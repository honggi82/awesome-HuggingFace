# arXiv:2511.22609v1[cs.CV]27Nov2025

## MG-Nav: Dual-Scale Visual Navigation via Sparse Spatial Memory

Bo Wang1,∗ Jiehong Lin1,∗ Chenzhi Liu1 Xinting Hu1 Yifei Yu1 Tianjia Liu1 Zhongrui Wang2,† Xiaojuan Qi1,†

wangzr@sustech.edu.cn xjqi@eee.hku.hk

1The University of Hong Kong 2Southern University of Science and Technology ∗Equal contribution †Corresponding authors

[Figure 1]

Figure 1. Overview of the proposed MG-Nav, a dual-scale framework that unifies global planning and local control for zero-shot visual navigation. (a) Global planning: MG-Nav plans over a Sparse Spatial Memory Graph (SMG), a compact region-centric memory that mirrors human navigation by providing node-level guidance without requiring dense 3D reconstruction. (b) Local navigation: MG-Nav employs navigation foundation policies enhanced with VGGT geometric features to improve goal recognition and enable obstacle-aware control. MG-Nav operates global planning and local navigation at different frequencies and uses periodic re-localization to correct errors, which effectively handles dynamic changes and avoids collisions compared to methods that rely on dense 3D reconstruction.

### Abstract

We present MG-Nav (Memory-Guided Navigation), a dual-scale framework for zero-shot visual navigation that unifies global memory-guided planning with local geometry-enhanced control. At its core is the Sparse Spatial Memory Graph (SMG), a compact, region-centric memory where each node aggregates multi-view keyframe and object semantics, capturing both appearance and spatial

structure while preserving viewpoint diversity. At the global level, the agent is localized on SMG and a goal-conditioned node path is planned via an image-to-instance hybrid retrieval, producing a sequence of reachable waypoints for long-horizon guidance. At the local level, a navigation foundation policy executes these waypoints in point-goal mode with obstacle-aware control, and switches to imagegoal mode when navigating from the final node towards the visual target. To further enhance viewpoint align-

ment and goal recognition, we introduce VGGT-adapter, a lightweight geometric module built on the pre-trained VGGT model, which aligns observation and goal features in a shared 3D-aware space. MG-Nav operates global planning and local control at different frequencies, using periodic re-localization to correct errors. Experiments on HM3D Instance-Image-Goal and MP3D Image-Goal benchmarks demonstrate that MG-Nav achieves state-ofthe-art zero-shot performance and remains robust under dynamic rearrangements and unseen scene conditions.

### 1. Introduction

Navigation is a fundamental capability for embodied agents to interact intelligently with their surroundings [18, 32]. From domestic robots and delivery drones to AR/VR telepresence systems, the ability to reach a visual target in previously unseen environments, without explicit maps or dense supervision, remains a key milestone toward general embodied intelligence. Within this context, visual navigation, where the goal is specified by an image depicting either a target object instance [11, 12, 36] or a scene [1, 17, 34], is particularly valuable yet challenging. The agent must infer the underlying 3D structure from a single goal image, reason over novel spatial layouts, and accurately reach the corresponding viewpoint or object while avoiding collisions and adapting to dynamic scene changes.

Existing frameworks for visual navigation can be broadly categorized into three families, each facing inherent limitations in unseen scenarios. (i) Foundation policy models [2, 29] leverage large-scale trajectory pretraining to achieve strong generalization, yet struggle when the goal is invisible, often degenerating into unguided exploration with weak long-range reasoning [25, 27]. (ii) Reinforcement Learning (RL) methods [15, 21, 30] excel in fine-grained control within known environments but demand massive interaction data and fail to generalize under domain shifts. (iii) Memory-based zero-shot approaches construct persistent global maps [9, 12, 24] or scene graphs [36] to enable long-horizon planning without retraining. However, these methods typically rely on dense RGB-D reconstructions, which are expensive to build and brittle to even mild rearrangements or dynamic changes after memory construction.

In light of these limitations, it is instructive to revisit how humans navigate, which offers a striking contrast and source of inspiration. Unlike artificial agents that depend on dense maps or precise metric reconstructions, humans navigate effectively through sparse visual memories, a limited set of distinctive snapshots that anchor our sense of place and orientation [5, 6, 28]. Such memories provide coarse yet reliable global guidance, enabling us to recall spatial relations and approximate locations even after long intervals or environmental changes. Concurrently, humans engage

in continuous local replanning, dynamically adapting motion to avoid obstacles, resolve occlusions, and respond to unforeseen scene variations [5, 6]. This hierarchical strategy, i.e., global guidance from sparse memory coupled with local reactive contron, forms the foundation of robust and adaptive navigation in complex, dynamic environments.

Inspired by this, we propose MG-Nav (Memory-Guided Navigation), a dual-scale framework that integrates global memory-guided planning with local geometry-enhanced control. Fig. 1 gives an overview of MG-Nav. At its core lies the Sparse Spatial Memory Graph (SMG), a compact, region-centric representation of the explored environment. Each node in the SMG corresponds to a spatial region and aggregates a small set of multi-view keyframes together with instance-level semantics, while edges encode navigable connectivity between regions. This sparse abstraction mirrors human navigation practice, capturing distinctive visual memories that preserve viewpoint diversity and semantic consistency without relying on dense 3D reconstruction.

At the global level, MG-Nav performs high-level planning over SMG. Given a visual goal, the agent is first localized on SMG through an image-to-instance hybrid retrieval mechanism that jointly matches both the current observation and the goal image to their corresponding nodes. Based on the matched nodes, a goal-conditioned node path is planned along the edges of SMG, yielding a sequence of reachable waypoints that provide global guidance and decompose the long-horizon navigation into node-to-node traversals.

At the local level, we employ a zero-shot foundation policy, pre-trained on large-scale data, which demonstrates strong obstacle avoidance and robustness to dynamic scene changes, to execute motion between adjacent SMG nodes. To further enhance geometric reasoning and visual goal alignment, we introduce VGGT-adapter, which incorporates geometry-aware features from the foundational Visual Geometry Group Transformer (VGGT) model [31] into the policy. This integration preserves spatial consistency under rapid viewpoint changes and improves precise alignment with the visual goal, particularly during the final approach.

Global planning and local control operate at different frequencies, with periodic re-localization and re-planning, to absorb execution noise and recover from errors. By seamlessly combining global node-level guidance with local, goal-directed control, MG-Nav achieves robust zeroshot visual navigation in complex, dynamic environments.

Extensive experiments show that MG-Nav attains stateof-the-art results on both HM3D Instance-Image-Goal and MP3D Image-Goal, reaching 78.5/59.3 and 83.8/57.1 (SR/SPL). Ablation studies verify the effectiveness of the different components of MG-Nav. Moreover, in dynamic environments with newly inserted obstacles, MG-Nav remains stable with only minor performance drops, demonstrating strong robustness.

Our main contributions are summarized as follows:

- • Dual-scale memory-guided navigation. We present MG-Nav, a unified architecture that integrates high-level semantic–topological planning with low-level geometryenhanced control, enabling robust zero-shot navigation.
- • Sparse Spatial Memory Graph (SMG). A regioncentric, multi-view memory graph representation that encodes both keyframe and object semantics, supporting efficient hybrid retrieval and node-level global planning.
- • Geometric enhancement for local navigation. We introduce a lightweight VGGT-adapter that aligns observation and goal embeddings in a shared 3D-aware space using VGGT features, substantially improving viewpoint robustness and goal matching precision.
- • Empirical results. MG-Nav achieves the state-of-theart zero-shot performance on HM3D/MP3D benchmarks, and maintains robustness under dynamic scene changes.

### 2. Related Work

Memory-Free Foundation Policy Methods. Foundation policies (e.g., GNM, ViNT, NoMaD, NavDP [2, 26, 27, 29]) provide reliable short-horizon, zero-shot control but remain memory-free, relying on reactive visual similarity without global state and leading to failures when the goal is out of view and weak long-range reasoning.

Memory-Free Reinforcement Learning Methods. Endto-end RL approaches [10, 15, 21, 30, 35] target finegrained control and incorporate LLMs for high-level command interpretation (e.g., CompassNav [14]). Yet they lack explicit global memory and often generalize poorly to unseen layouts and object configurations, struggling to maintain instance–image goal consistency over long horizons.

Memory-Based Methods. For long-range planning, memory-based methods build global scene representations. Metric or dense maps (e.g., BSC-Nav, GOAT, IEVE, MODIIN [9, 11, 12, 24]) and neural volumetric models (e.g., GaussNav, GSplatVNM [8, 13]) offer accurate pose fidelity but are heavy and brittle under rearrangements. Sparse graphs are more efficient, yet object-centric graphs (e.g., UniGoal [36]) lose local context, while topology-heavy systems (e.g., Astra, Mobility-VLN [4, 33]) add complexity by intertwining graph reasoning with LLM-based planning. We address these trade-offs with a sparse, regioncentric Scene Memory Graph (SMG) that preserves local context for hybrid retrieval and a dual-scale design combining SMG-based global planning with robust local control.

### 3. Memory-Guided Navigation

##### 3.1. Overview

Visual navigation aims to predict a sequence of M actions A = (a1,...,aM) that guide an agent to a visual goal Igoal depicting the target object or scene, where each ac-

tion is defined by planar displacement and orientation. To address this task, we propose MG-Nav (Memory-Guided Navigation), a dual-scale framework that integrates global planning with local execution.

An overview of the MG-Nav workflow is shown in Fig.

- 2. The core idea is that a small number of distinctive visual memories provide coarse global guidance, while local motion is continuously adapted to avoid obstacles and dynamic changes. To embody this idea, we introduce Sparse Spatial Memory Graph (SMG) as an abstract scene representation (Sec. 3.2). Formally, SMG is defined as G = (V,E), where nodes V function as sparse, memorable spatial anchors, and edges E encode their navigable connectivity. Upon G, MGNav performs dual-scale navigation: (i) global planning, which retrieves the memory node vgoal ∈ V best aligned with the goal image Igoal and plans a node-level path (v1,...,vK−1,vK) along E (Sec. 3.3), where vK corresponds to the exact visual goal Igoal with vK−1 = vgoal; and (ii) local navigation, where a geometry-enhanced diffusion policy navigates between consecutive nodes to consti-

tute the whole action sequence A = (A1,...,AK−1,AK) (Sec. 3.4). By operating global planning and local navigation at different frequencies, with periodic re-localization and re-planning, MG-Nav mitigates execution noise and drift, and thus enables robust zero-shot navigation in dynamic and previously unseen environments (Sec. 3.5).

- 3.2. Sparse Spatial Memory Graph

Sparse Spatial Memory Graph (SMG) G = (V,E) forms the core of MG-Nav to enable global planning. The construction process of G is illustrated in Fig. 3.

More specifically, for each indoor scene, we first follow standard practice to collect posed tour demonstrations [13, 24]. We then apply Farthest-Point Sampling (FPS) to the extrinsic camera poses of the demonstration frames to obtain sparse but spatially representative locations. Around each sampled location, we define a region of radius r as a spatial node v ∈ V, while the edge set E is derived from temporal adjacency during the tours, which naturally ensures that connected nodes correspond to mutually reachable regions in the physical environment.

We further represent each spatial node as a structural embedding v = (P,F,O), where P ∈ R3 denotes the 3D center of the region, F ∈ RN

f×C contains Nf keyframe embeddings, and O ∈ RN

o×C contains No representative object embeddings, with C as the feature dimension. Concretely, P is set as the sampled camera location of v, while F and O are obtained as follows:

(i) Keyframe embeddings - We extract the CLS token from DINOv2 [19] for demonstration frames within the local region of v. To ensure diversity, we first select Nf′ frames with the lowest feature similarity and then pick the final Nf frames with the largest camera rotation variance

(a) Sparse Spaital Memory Graph 𝓖(𝓥,𝓔)

[Figure 2]

[Figure 3]

|[Figure 4]<br><br>Edge 𝓔 Node 𝓥 Key Frame<br><br>[Figure 5]<br><br>[Figure 6]<br><br>[Figure 7]<br><br>[Figure 8]<br><br>KeyFrame Embedding<br><br>Object Embedding<br><br>Plant 1<br><br>DINOv2<br><br>Chair 1 Chair 2<br><br>[Figure 9]<br><br>[Figure 10]<br><br>[Figure 11]<br><br>[Figure 12]|
|---|

Exploration

Observation Agent Environment

|[Figure 13]<br><br>[Figure 14]<br><br>[Figure 15]<br><br>RGB Cam. Pose<br><br>[Figure 16]<br><br>6-DoF (𝑿,𝒀,𝒁,∅)|
|---|

||[Figure 17]| |
|---|---|
| | |
<br><br>|[Figure 18]| |
|---|---|
| | |
<br><br>Observation t Goal<br><br>Image-Node Matching<br><br>Keyframe Retrieval<br><br>Object Retrieval<br><br>M<br><br>Node i Match Si<br><br>[Figure 19]<br><br>Sofa<br><br>Table<br><br>Object Retrieval<br><br>Sofa 1:<br><br>Sofa 2:<br><br><br>Table 1:<br><br>Table 2:<br><br><br>Goal Object Node i Object<br><br>M Sobj<br><br>Mean<br><br>M<br><br>Keyframe Retrieval<br><br>[Figure 20]<br><br>[Figure 21]<br><br>[Figure 22]<br><br>[Figure 23]<br><br>[Figure 24]<br><br>M Skey<br><br>Goal Scene Node i Scene Mean<br><br>M<br><br>|[Figure 25]|
|---|
<br><br>Matched Observation t Matched Goal<br><br>Node-Level Path<br><br>f2|
|---|

|[Figure 26]<br><br>|Foundation Model|
|---|
<br><br>Generated Trajectories<br><br>|Geo-Foundation Model|
|---|
<br><br>[Figure 27]<br><br>[Figure 28]<br><br>Geometry Match<br><br>[Figure 29]<br><br>[Figure 30]<br><br>Obstacle Avoidance<br><br>[Figure 31]<br><br>Move toward the goal while avoiding obstacles<br><br>VGGT Enhance<br><br>f1|
|---|

Asynchronous

(b) Global Planning with SMG (c) Local Navigation

- Figure 2. Illustration of the navigation process of MG-Nav, a dual-scale framework combining global planning with local execution. (a) Sparse Spatial Memory Graph (SMG) serves as a compact, region-centric memory; each node aggregates multi-view keyframes and object semantics, while edges encode navigable connectivity. (b) Global Planning with SMG: Both the agent and the goal are localized on the SMG via an image-to-instance hybrid node retrieval. A goal-conditioned path from the current observation at time t to the goal is then planned along the graph edges to provide global guidance. (c) Local Navigation via Geometry-Enhanced Policy: a navigation foundation policy, geometrically enhanced with the VGGT-adapter, moves the agent between adjacent nodes while maintaining obstacle

avoidance and accurate visual goal alignment. By running global planning (f1) and local navigation (f2) at different frequencies with periodic re-localization, MG-Nav achieves robust zero-shot navigation in dynamic, unseen environments.

among them. The embeddings of these selected frames form F, providing rich regional context.

visual memory anchors to simplify the planning space.

##### 3.3. Global Planning with SMG

(ii) Object embeddings - We apply the open-vocabulary segmentation model Grounded-SAM [23] to all frames within the region of v to extract object instances. For each segmented object, we use DINOv2 to obtain embeddings from the CLS token and the average patch features, which are combined via a weighted summation to form robust object embeddings. Objects of the same category whose embedding cosine similarity exceeds a threshold τ are further considered the same instance, and their embeddings are averaged to produce a unique feature in O.

As shown in Fig. 2, global planning on SMG G = (V,E) consists of two steps, including (i) matching both the current observation Iobs and the visual goal Igoal to nodes in V, and (ii) planning a node-level trajectory along SMG edges that provides high-level guidance for local navigation.

###### 3.3.1. Image-Node Matching

Given an input image (Iobs or Igoal), we aim to retrieve the node in V most semantically and visually aligned with it, using a hybrid retrieval strategy combining global scene similarity and object-level semantics:

SMG provides a sparse yet spatially representative structure that integrates information from both multi-view keyframes and object instances, thus preserving distinctive

(i) Keyframe retrieval - For each node v ∈ V, we compute the cosine similarity between the DINOv2 CLS embed-

[Figure 32]

viating the difficulty of fine-grained motion control toward the final visual goal. Building upon this, we further address the zero-shot local navigation between adjacent nodes using general navigation foundation policies, and introduce VGGT-adapter for geometric enhancement to improve visual goal matching capacity of policies.

###### Exploration Tour

###### FPS

###### Temporal Connect

Node Region Node Center

Temporally Adjacent Temporally Distant

Tour Path Observation

[Figure 33]

[Figure 34]

Node Edge

|Feature Sim.<br><br>[Figure 35]<br><br>[Figure 36]<br><br>𝜽𝒎𝒂𝒙<br><br>Camera Rot.<br><br>Max Camera Yaw<br><br>[Figure 37]<br><br>[Figure 38]<br><br>[Figure 39]<br><br>[Figure 40]<br><br>Keyframe<br><br>|
|---|

|[Figure 41]<br><br>[Figure 42]<br><br>…<br><br>Segmentation<br><br>||> 𝝉|
|---|
| |
|---|---|
| | |
|> 𝝉| |
| | |
| | |
<br><br>Table 2 Fusion Embedding<br><br>Table 1 Fusion Embedding<br><br>Object Embedding|
|---|

- 3.4.1. Geometry-Enhanced Policy with VGGT-adapter Navigation foundation policies are typically formulated as A = π(Iobs,Pgoal,Igoal), where the goal is specified either as a point Pgoal or an image Igoal. Although these policies are trained on large-scale trajectories and demonstrate strong short-horizon navigation capabilities with obstacle avoidance and robustness to dynamic scene changes, they still struggle to precisely localize and pursue visual goals, especially under significant viewpoint variations.

To mitigate this limitation, we introduce the VGGTadapter, which integrates geometry-aware features derived from the pre-trained Visual Geometry Group Transformer (VGGT) model [31] into the policy. VGGT inherently models multi-view 3D structure and pixel correspondences, making it suitable for enhancing spatial reasoning and improving visual goal matching during navigation.

More specifically, we use VGGT to extract geometryaware feature maps from both the current observation Iobs and the visual goal Igoal. After applying spatial average pooling, the resulting feature maps are flattened into token sequences FGobs and FGgoal. The two token sequences are then concatenated and passed through the VGGTadapter, implemented as a lightweight multi-layer perceptron (MLP), which projects the geometry tokens into the policy embedding space as FGadpt = MLP([FGobs |FGgoal]). Finally, FGadpt is concatenated with the original visual condition tokens of the navigation policy, thereby enriching it with 3D geometric cues and inter-frame correspondence awareness to strengthen visual goal matching.

- 3.4.2. Node-based Local Navigation

###### …

Keyframe Selection Object Embedding

- Figure 3. Illustration of the construction of Sparse Spatial Memory Graph. Each node in SMG represents a spatial region, aggregating a small set of both multi-view keyframe and object embeddings, while edges between nodes encode navigable connectivity.

ding of the input image and each of the Nf keyframe embeddings F, taking the maximum similarity as the keyframe score of the node. The top-No nodes with the highest keyframe scores are retained as candidates for subsequent object-level verification.

(ii) Object retrieval - Following the procedure in Sec.

- 3.2, we apply Grounded-SAM for open-vocabulary segmentation and DINOv2 for feature extraction to obtain object embeddings of the image. For each object embedding in the image, we compute its cosine similarity with object embeddings of the same category in each node candidate, taking the maximum score; if a node does not contain that category, the score is set to 0. For each node, we average the scores across all objects to obtain its object similarity score.

Finally, for the top-No node candidates, we average their keyframe and object similarity scores, and select the node with the highest combined score as the retrieval result.

- 3.3.2. Node-Level Path Planning Based on the hybrid retrieval results, we obtain the matched

nodes vobs and vgoal corresponding to Iobs and Igoal. We then perform high-level planning on SMG G by applying A* search [16] over the edge set E. Since each edge denotes a physically feasible transition observed in demonstrations, the resulting path provides a reliable structural prior for navigation. The planned node-level trajectory is denoted as (v1,...,vK−1,vK), where v1 = vobs and vK−1 = vgoal; we additionally append vK to represent Igoal, ensuring that the node sequence explicitly terminates at the goal state.

- 3.4. Local Navigation via Geometry-Enhanced Policy

With the node-level path {v1,...,vK} obtained from SMG, our geometric-enhanced policy π is employed to perform local navigation in a node-to-node manner to progressively approach the final visual goal while avoiding obstacles. For each step k, the agent navigates from the current node vk−1 to the next node vk by predicting a chunk of executable actions:

π(Ik,Pk,−) k ̸= K π(Ik,−,Igoal) k = K

, (1)

Ak =

where Ik denotes the current observation and Pk is the 3D location of node vk. Accordingly, the policy operates in point-goal mode when moving between intermediate nodes, and switches to image-goal mode only at the final stage to precisely align with the visual goal Igoal, befitting from the geometric enhancement of VGGT-adapter.

The node-level path planned on SMG decomposes navigation into a sequence of intermediate sub-goals, thereby alle-

The overall navigation action sequence is thus given as A = (A1,...,AK−1,AK).

##### 3.5. Dual-scale Planning Loop

MG-Nav coordinates planning and control at two frequencies: a slow global loop (every Tg steps) and a fast local loop (every Tℓ ≪ Tg). The global loop re-localizes the agent on SMG via hybrid retrieval and updates the node-level path (Sec. 3.3) whenever confidence drops or edges become blocked, yielding an updated node sequence. The local loop continuously executes short-horizon actions toward the current node while performing obstacle-aware control (Sec. 3.4). At runtime, MG-Nav alternates between two tightly coupled processes:

- • Local navigation (Sec. 3.4): The geometry-enhanced

policy executes Ak to reach the current node vk, updating visual feedback at each step.

- • Periodic global re-localization and planning (Sec. 3.3):

Every Tg local steps—or when the visual-confidence score drops—the agent is re-localized on SMG and A* is re-invoked to refine the remaining path.

This asynchronous, dual-frequency scheme mitigates drift, adapts to dynamic scene changes, and preserves longhorizon goal intent with minimal computational overhead.

### 4. Experiments

##### 4.1. Experimental Setup

Benchmarks: Our method is tested on two 3D scene datasets, Habitat-Matterport 3D (HM3D) [22] for instance image-goal navigation (InstanceImageNav) and Matterport3D (MP3D) [3] for image-goal navigation (ImageNav). Both tasks require the agent to navigate toward the target depicted in a given goal image, but ImageNav targets an entire scene or viewpoint, while InstanceImageNav specifies a particular object instance. For HM3D, we conduct experiments on 1000 episodes of 36 validation scenes following the Instance ImageNav-v3. For MP3D ImageNav, we conduct experiments on 1014 episodes of 5 testing scenes.

Evaluation Metrics: We evaluate with Success Rate (SR) to measure the proportion of successful episodes, and Success Rate weighted by Path Length (SPL) [11, 36] to assess navigation efficiency by accounting for path optimality.

Implementation Details: Our agent is deployed in Habitat Simulator [20]. We employ DINOv2-ViT-L/14 [19] for visual encoding, Grounded-SAM-2 [23] for semantic segmentation, and VGGT [31] for geometric embedding extraction. We use NavDP [2] as the foundation policy for MG-Nav’s local navigation module. The success distance threshold is 1.0 m and the maximum episode length is 500 steps. Hyperparameters are detailed in appendix.

- Table 1. InstanceImageNav on HM3D. Foundation models are shown on top, RL-based methods are shown on middle, and Memory-based methods are shown on bottom.

Method Memory SR ↑ SPL ↑

ViNT* [27] – 7.7 6.7 GNM* [26] – 11.4 5.6 NoMAD* [29] – 16.8 7.0 NavDP* [2] – 24.7 12.6

RL Baseline† [10] – 8.3 3.5 FGPrompt† [30] – 9.9 2.8 OVRL-v2 IIN† [35] – 24.8 11.8 CompassNav [14] – 35.6 14.8

GOAT [9] Scene Map (RGBD) 37.4 16.1 MOD-IIN [11] Scene Map (RGBD) 56.1 23.3 UniGoal [36] Object Graph (RGBD) 60.2 23.7 IEVE [12] Scene Map (RGBD) 70.2 25.2 BSC-Nav [24] Scene Map (RGBD) 71.4 57.2 GaussNav [13] 3DGS Map (RGBD) 72.5 57.8 MG-Nav(Ours) SMG (RGB) 78.5 59.3

* Result from our re-implementation following their official code. † Result from GaussNav.

- Table 2. ImageNav on MP3D. Foundation models are shown on top, RL-based methods are shown on middle, and Memory-based methods are shown on bottom.

###### Method Memory SR ↑ SPL ↑

ViNT* [27] – 6.12 4.98 GNM* [26] – 10.06 7.36 NoMAD* [29] – 10.95 5.64 NavDP* [2] – 15.49 7.96

RSFG* [7] – 68.54 43.18 FGPrompt-MF* [30] – 71.20 39.74 REGNav* [15] – 74.66 47.24 FGPrompt-EF* [30] – 77.71 51.09

###### MG-Nav(Ours) SMG (RGB) 83.77 57.15

* Result from our re-implementation following their official code.

##### 4.2. Main Results

HM3D InstanceImageNav. Table 1 reports HM3D InstanceImageNav results. Foundation policies (e.g., NavDP [2], 24.7/12.6) and RL methods (e.g., CompassNav [14], 35.6/14.8) suffer from poor zero-shot generalization due to reactive local planning and weak global reasoning. While Memory-based methods (e.g., BSC-Nav [24], 71.4/57.2; GaussNav [13], 72.5/57.8) achieve notable accuracy, they remain limited by maps that lack robust instance seman-

[Figure 43]

###### Stage 2: Point Nav and Relocal. Stage 3: Image Nav and Verified

###### Goal and Overall Navigation

###### Stage 1: Local. and Path Plan

|Step 164|
|---|

|Step 140|
|---|

|Step 1|
|---|

|Step 73|
|---|

|[Figure 44]<br><br>[Figure 45]|
|---|

|[Figure 46]|
|---|

[Figure 47]

|[Figure 48]<br><br>[Figure 49]|
|---|

|[Figure 50]<br><br>[Figure 51]|
|---|

|[Figure 52]<br><br>[Figure 53]|
|---|

|[Figure 54]<br><br>[Figure 55]|
|---|

|[Figure 56]|
|---|

|[Figure 57]<br><br>[Figure 58]|
|---|

Image Goal

Scene

Step 73 Image

Point Nav.

Point Nav.

Image Nav.

Step 1 Image Goal Image

[Figure 59]

Shortest Path

[Figure 60]

[Figure 61]

[Figure 62]

Navigation Path Goal Region Graph Node Graph Edge Diffusion Traj.

###### A* Path Replan

###### Short trajectory, verify

A* Path based Graph

Goal Local.

Self Local.

Self Local.

Memory Graph and Navigation

- Figure 4. Illustration of the decision process of MG-Nav. Step 1 shows initial self- and goal-localization on SMG followed by global planning. Step 73 shows node-to-node navigation using point mode of NavDP, with periodic global re-localization. Step 140 shows the agent entering the matched goal-node region. Step 164 shows the policy switching to image mode and successfully verifying the target.

SR/SPL). We perform an ablation to study the necessity of each component. Relying only on global keyframe retrieval degrades performance by a large margin (from 78.50/59.27 to 73.52/52.90), and using only object-level retrieval also results in a substantial drop (to 72.20/52.79). This confirms the need for a hybrid strategy. Global appearance enables coarse scene localization and long-range recall, while instance semantics enhance object-level discrimination and viewpoint invariance. Combining both yields more reliable and robust matching.

tics and rely on unstable low-level planners. MG-Nav addresses these limitations by pairing a sparse, region-centric SMG with a foundation policy enhanced by a VGGT-based adapter for robust navigation and viewpoint alignment. This synthesis establishes a new state-of-the-art of SR 78.5 and yields shorter, more reliable paths (SPL 59.3). The successful example of the task executed by MG-Nav is shown in Fig. 4, and other examples are detailed in appendix.

MP3D ImageNav. Table 2 reports overall MP3D ImageNav results. MG-Nav attains 83.77/57.15 (SR/SPL), surpassing the strongest RL baseline FGPrompt-EF [30] 77.71/51.09 by 6.06/6.06. Foundation policies remain far lower, for example, NavDP [2] performs 15.49 and 7.96, where our method surpasses by 68.28/49.19. The huge improvement demonstrates that our dual-scale navigation system via SMG provide reliable global planning and robust view alignment, yielding state-of-the-art performance.

Graph sparsity ablation. We analyze the impact of the SMG structure by adjusting node spacing (d) and coverage radius (r). The configuration (d=1.0m,r=0.5m) achieves the best 78.50/59.27 (SR/SPL). Making the graph moderately sparser to (1.5m,0.8m) yields 77.10/54.79 (−1.40/−4.48). Further sparsifying to (2.0m,1.0m) degrades to 70.69/46.89 (−7.81/−12.38). The sharper decline in SPL indicates that overly sparse graphs reduce viewpoint diversity and weaken topological connectivity, leading to longer local execution segments and detours.

##### 4.3. Ablation Study

Component ablation. Table 3 reports a detailed ablation on model components. The foundation model achieves only 24.7/12.6 (SR/SPL), limited by its purely reactive control and lack of long-range planning. Introducing SMG for global planning significantly boosts performance to 74.04/56.14, as it decomposes long-range navigation into reachable node-to-node subgoals, providing global guidance for the agent. Further incorporating the VGGT-adapter improves results to 78.50/59.27, confirming that geometryaware observation–goal alignment enhances robustness during viewpoint shifts and final approach stages.

##### 4.4. Robustness to Dynamic Scene Changes

To assess robustness to unmodeled dynamic scene changes, we first construct the Memory Graph on the original HM3D scenes. We then insert different numbers (0/5/10) of random obstacles into the environment during the navigation phase to simulate a dynamic environment (Fig. 5). These newly inserted objects serve as dynamic obstacles that the agent must robustly avoid to reach its goal. The performance of MG-Nav under these conditions is then compared with representative mapping-based methods (Table 4). We observe that methods degrade sharply (e.g., BSC-Nav [24] SR 25.49 → 7.84, UniGoal [36] SR 56.43 → 44.2), while MG-Nav shows only minor drops (SR 73.53 → 68.63 and SPL 56.28

Retrieval strategy ablation. Our hybrid retrieval strategy, which combines keyframe (global) and object-level semantics, achieves the best performance (78.50/59.27 in

- Table 3. Ablation study of our method. We report SR/SPL while jointly exposing graph sparsity (node spacing d) and coverage radius r. “Global” means keyframe retrieval, “Object” means object retrieval.

Ablation Variant Graph Retrieval VGGT-adapter d (m) r (m) SR ↑ / SPL ↑

Component

Foundation Policy (NavDP) × – × – – 24.70 / 12.60 + SMG ✓ Global+Object × 1.0 0.5 74.04 / 56.14 + VGGT-adapter ✓ Global+Object ✓ 1.0 0.5 78.50 / 59.27

Retrieval

Instance Match only ✓ Object ✓ 1.0 0.5 72.20 / 52.79

Global Match only ✓ Global ✓ 1.0 0.5 73.52 / 52.90 Global + Instance Match ✓ Global+Object ✓ 1.0 0.5 78.50 / 59.27

Graph sparsity

- Graph 1 ✓ Global+Object ✓ 2.0 1.0 70.69 / 46.89
- Graph 2 ✓ Global+Object ✓ 1.5 0.8 77.10 / 54.79
- Graph 3 ✓ Global+Object ✓ 1.0 0.5 78.50 / 59.27

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

|[Figure 67]|
|---|

[Figure 68]

|[Figure 69]|
|---|

Step=88

Step=94

Step=78

Step=632

Step=883

Step=159

|[Figure 70]|
|---|

|[Figure 71]|
|---|

|[Figure 72]|
|---|

|[Figure 73]|
|---|

[Figure 74]

|[Figure 75]|
|---|

Obstacle MG-Nav Path

[Figure 76]

MG-Nav UniGoal

|[Figure 77]|
|---|

Success

|[Figure 78]|
|---|

Fail

[Figure 79]

[Figure 80]

Original Scene (memory build) Dynamic Scene (with obstacles)

Start Final UniGoal Path

[Figure 81]

[Figure 82]

|[Figure 83]|
|---|

Goal Image

[Figure 84]

[Figure 85]

[Figure 86]

(Success) (Fail)

|[Figure 87]|
|---|

|[Figure 88]|
|---|

|[Figure 89]|
|---|

Dynamic Scene (with obstacles)

Figure 5. Illustration of the robustness to dynamic scene changes of MG-Nav and UniGoal. 10 additional obstacles are added to scene mv2HUxq3B53 (left) to model dynamic scenarios (middle). UniGoal becomes trapped near the inserted obstacles and keeps wandering in a local region until timeout (right, green path), whereas MG-Nav successfully avoids the newly added obstacles and reaches the goal (right, red path), demonstrating strong robustness to unmodeled scene rearrangements.

- Table 4. Robustness to dynamic scene changes. We evaluate each model on 100 InstanceImageNav episodes from HM3D under our dynamic-scene setting. Results for UniGoal and BSC-Nav are obtained from our re-implementations based on their official code.

### 5. Conclusion

We introduced MG-Nav, a dual-scale navigation framework that combines global memory-guided planning with local geometry-enhanced control for zero-shot visual navigation. The core Sparse Spatial Memory Graph (SMG) provides a compact, region-centric representation, enabling long-horizon reasoning and robustness to moderate scene changes. Global planning retrieves goal-conditioned node paths, while a pre-trained navigation policy, augmented with the VGGT-adapter, executes node-to-node motion with enhanced visual and geometric alignment. Experiments show that MG-Nav achieves state-of-the-art zero-shot performance on challenging Image-Goal and Instance-Goal tasks, generalizing effectively to novel and dynamic environments. We believe that the dual-scale navigation with sparse spatial memory can inspire future research in scalable and robust embodied navigation, especially for agents operating in complex, unseen, or dynamic spaces.

Obs. Num 0 5 10 Method SR SPL SR SPL SR SPL BSC-Nav [24] 25.49 19.91 8.64 4.63 7.84 4.94 UniGoal [36] 56.43 20.44 52.94 19.68 44.21 17.17 Ours 73.53 56.28 72.55 52.20 68.63 50.15

→ 50.15). This resilience stems from a decoupled dualscale design. The sparse SMG delivers robust region-level global planning, keeping the navigational goal stable, while the zero-shot local policy handles unmodeled obstacles and avoids them without relying on the global map, resulting in only minor drops in SR and SPL.

### References

- [1] Ziad Al-Halah, Santhosh Kumar Ramakrishnan, and Kristen Grauman. Zero experience required: Plug & play modular transfer learning for semantic visual navigation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 17031–17041, 2022. 2
- [2] Wenzhe Cai, Jiaqi Peng, Yuqiang Yang, Yujian Zhang, Meng Wei, Hanqing Wang, Yilun Chen, Tai Wang, and Jiangmiao Pang. Navdp: Learning sim-to-real navigation diffusion policy with privileged information guidance. arXiv preprint arXiv:2505.08712, 2025. 2, 3, 6, 7
- [3] Angel Chang, Angela Dai, Thomas Funkhouser, Maciej Halber, Matthias Niessner, Manolis Savva, Shuran Song, Andy Zeng, and Yinda Zhang. Matterport3d: Learning from rgbd data in indoor environments. International Conference on 3D Vision (3DV), 2017. 6
- [4] Sheng Chen, Peiyu He, Jiaxin Hu, Ziyang Liu, Yansheng Wang, Tao Xu, Chi Zhang, Chongchong Zhang, Chao An, Shiyu Cai, et al. Astra: Toward general-purpose mobile robots via hierarchical multimodal learning. arXiv preprint arXiv:2506.06205, 2025. 3
- [5] Arne D Ekstrom and Paul F Hill. Spatial navigation and memory: A review of the similarities and differences relevant to brain models and age. Neuron, 111(7):1037–1049,

2023. 2

- [6] Russell A Epstein, Eva Zita Patai, Joshua B Julian, and Hugo J Spiers. The cognitive map in humans: spatial navigation and beyond. Nature neuroscience, 20(11):1504–1513,

2017. 2

- [7] Zhicheng Feng, Xieyuanli Chen, Chenghao Shi, Lun Luo, Zhichao Chen, Yun-Hui Liu, and Huimin Lu. Image-goal navigation using refined feature guidance and scene graph enhancement. arXiv preprint arXiv:2503.10986, 2025. 6
- [8] Kohei Honda, Takeshi Ishita, Yasuhiro Yoshimura, and Ryo Yonetani. Gsplatvnm: Point-of-view synthesis for visual navigation models using gaussian splatting. arXiv preprint arXiv:2503.05152, 2025. 3
- [9] Mukul Khanna, Ram Ramrakhya, Gunjan Chhablani, Sriram Yenamandra, Theophile Gervet, Matthew Chang, Zsolt Kira, Devendra Singh Chaplot, Dhruv Batra, and Roozbeh Mottaghi. Goat-bench: A benchmark for multi-modal lifelong navigation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 16373– 16383, 2024. 2, 3, 6
- [10] Jacob Krantz, Stefan Lee, Jitendra Malik, Dhruv Batra, and Devendra Singh Chaplot. Instance-specific image goal navigation: Training embodied agents to find object instances. arXiv preprint arXiv:2211.15876, 2022. 3, 6
- [11] Jacob Krantz, Theophile Gervet, Karmesh Yadav, Austin Wang, Chris Paxton, Roozbeh Mottaghi, Dhruv Batra, Jitendra Malik, Stefan Lee, and Devendra Singh Chaplot. Navigating to objects specified by images. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 10916–10925, 2023. 2, 3, 6
- [12] Xiaohan Lei, Min Wang, Wengang Zhou, Li Li, and Houqiang Li. Instance-aware exploration-verificationexploitation for instance imagegoal navigation. In Proceed-

- ings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 16329–16339, 2024. 2, 3, 6
- [13] Xiaohan Lei, Min Wang, Wengang Zhou, and Houqiang Li. Gaussnav: Gaussian splatting for visual navigation. IEEE Transactions on Pattern Analysis and Machine Intelligence,

2025. 3, 6

- [14] LinFeng Li, Jian Zhao, Yuan Xie, Xin Tan, and Xuelong Li. Compassnav: Steering from path imitation to decision understanding in navigation. arXiv preprint arXiv:2510.10154,

2025. 3, 6

- [15] Pengna Li, Kangyi Wu, Jingwen Fu, and Sanping Zhou. Regnav: Room expert guided image-goal navigation. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 4860–4868, 2025. 2, 3, 6
- [16] Xiang Liu and Daoxiong Gong. A comparative study of astar algorithms for search and rescue in perfect maze. In 2011 international conference on electric information and control engineering, pages 24–27. IEEE, 2011. 5
- [17] Lina Mezghan, Sainbayar Sukhbaatar, Thibaut Lavril, Oleksandr Maksymets, Dhruv Batra, Piotr Bojanowski, and Karteek Alahari. Memory-augmented reinforcement learning for image-goal navigation. In 2022 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 3316–3323. IEEE, 2022. 2
- [18] Piotr Mirowski, Razvan Pascanu, Fabio Viola, Hubert Soyer, Andrew J Ballard, Andrea Banino, Misha Denil, Ross Goroshin, Laurent Sifre, Koray Kavukcuoglu, et al. Learning to navigate in complex environments. arXiv preprint arXiv:1611.03673, 2016. 2
- [19] Maxime Oquab, Timoth´ee Darcet, Theo Moutakanni, Huy V. Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, Russell Howes, Po-Yao Huang, Hu Xu, Vasu Sharma, ShangWen Li, Wojciech Galuba, Mike Rabbat, Mido Assran, Nicolas Ballas, Gabriel Synnaeve, Ishan Misra, Herve Jegou, Julien Mairal, Patrick Labatut, Armand Joulin, and Piotr Bojanowski. Dinov2: Learning robust visual features without supervision, 2023. 3, 6
- [20] Xavi Puig, Eric Undersander, Andrew Szot, Mikael-Dallaire Cote, Ruslan Partsey, Jimmy Yang, Ruta Desai, Alexander William Clegg, Michal Hlavac, Tiffany Min, Theo Gervet, Vladim´ır Vondruˇs, Vincent-Pierre Berges, John Turner, Oleksandr Maksymets, Zsolt Kira, Mrinal Kalakrishnan, Jitendra Malik, Devendra Singh Chaplot, Unnat Jain, Dhruv Batra, Akshara Rai, and Roozbeh Mottaghi. Habitat 3.0: A co-habitat for humans, avatars and robots, 2023. 6
- [21] Zheng Qin, Le Wang, Yabing Wang, Sanping Zhou, Gang Hua, and Wei Tang. Rsrnav: Reasoning spatial relationship for image-goal navigation. arXiv preprint arXiv:2504.17991,

2025. 2, 3

- [22] Santhosh Kumar Ramakrishnan, Aaron Gokaslan, Erik Wijmans, Oleksandr Maksymets, Alexander Clegg, John M Turner, Eric Undersander, Wojciech Galuba, Andrew Westbury, Angel X Chang, Manolis Savva, Yili Zhao, and Dhruv Batra. Habitat-matterport 3d dataset (HM3d): 1000 largescale 3d environments for embodied AI. In Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2), 2021. 6

- [23] Tianhe Ren, Shilong Liu, Ailing Zeng, Jing Lin, Kunchang Li, He Cao, Jiayu Chen, Xinyu Huang, Yukang Chen, Feng Yan, Zhaoyang Zeng, Hao Zhang, Feng Li, Jie Yang, Hongyang Li, Qing Jiang, and Lei Zhang. Grounded sam: Assembling open-world models for diverse visual tasks,

2024. 4, 6

- [24] Shouwei Ruan, Liyuan Wang, Caixin Kang, Qihui Zhu, Songming Liu, Xingxing Wei, and Hang Su. From reactive to cognitive: brain-inspired spatial intelligence for embodied agents. arXiv preprint arXiv:2508.17198, 2025. 2, 3, 6, 7, 8
- [25] Matt Schmittle, Rohan Baijal, Nathan Hatch, Rosario Scalise, Mateo Guaman Castro, Sidharth Talia, Khimya Khetarpal, Byron Boots, and Siddhartha Srinivasa. Long range navigator (lrn): Extending robot planning horizons beyond metric maps. arXiv preprint arXiv:2504.13149, 2025. 2
- [26] Dhruv Shah, Ajay Sridhar, Arjun Bhorkar, Noriaki Hirose, and Sergey Levine. Gnm: A general navigation model to drive any robot. arXiv preprint arXiv:2210.03370, 2022. 3, 6
- [27] Dhruv Shah, Ajay Sridhar, Nitish Dashora, Kyle Stachowicz, Kevin Black, Noriaki Hirose, and Sergey Levine. Vint: A foundation model for visual navigation. arXiv preprint arXiv:2306.14846, 2023. 2, 3, 6
- [28] Alexander W Siegel and Sheldon H White. The development of spatial representations of large-scale environments. Advances in child development and behavior, 10:9–55, 1975. 2
- [29] Ajay Sridhar, Dhruv Shah, Catherine Glossop, and Sergey Levine. Nomad: Goal masked diffusion policies for navigation and exploration. In 2024 IEEE International Conference on Robotics and Automation (ICRA), pages 63–70. IEEE, 2024. 2, 3, 6
- [30] Xinyu Sun, Peihao Chen, Jugang Fan, Jian Chen, Thomas Li, and Mingkui Tan. Fgprompt: fine-grained goal prompting for image-goal navigation. Advances in Neural Information Processing Systems, 36:12054–12073, 2023. 2, 3, 6, 7
- [31] Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea Vedaldi, Christian Rupprecht, and David Novotny. Vggt: Visual geometry grounded transformer. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 5294–5306, 2025. 2, 5, 6
- [32] Lik Hang Kenny Wong, Xueyang Kang, Kaixin Bai, and Jianwei Zhang. A survey of robotic navigation and manipulation with physics simulators in the era of embodied ai. arXiv preprint arXiv:2505.01458, 2025. 2
- [33] Zhuo Xu, Hao-Tien Lewis Chiang, Zipeng Fu, Mithun George Jacob, Tingnan Zhang, Tsang-Wei Edward Lee, Wenhao Yu, Connor Schenck, David Rendleman, Dhruv Shah, et al. Mobility vla: Multimodal instruction navigation with long-context vlms and topological graphs. In 8th Annual Conference on Robot Learning, 2024. 3
- [34] Karmesh Yadav, Ram Ramrakhya, Arjun Majumdar, Vincent-Pierre Berges, Sachit Kuhar, Dhruv Batra, Alexei Baevski, and Oleksandr Maksymets. Offline visual representation learning for embodied navigation. In Workshop on Reincarnating Reinforcement Learning at ICLR 2023, 2023. 2

- [35] Karmesh Yadav, Ram Ramrakhya, Santhosh Kumar Ramakrishnan, Theo Gervet, John Turner, Aaron Gokaslan, Noah Maestre, Angel Xuan Chang, Dhruv Batra, Manolis Savva, et al. Habitat-matterport 3d semantics dataset. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4927–4936, 2023. 3, 6
- [36] Hang Yin, Xiuwei Xu, Linqing Zhao, Ziwei Wang, Jie Zhou, and Jiwen Lu. Unigoal: Towards universal zero-shot goaloriented navigation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 19057–19066,

2025. 2, 3, 6, 7, 8

