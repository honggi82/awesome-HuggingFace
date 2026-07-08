# arXiv:2604.00528v2[cs.CV]2Apr2026

## Think, Act, Build: An Agentic Framework with Vision Language Models for Zero-Shot 3D Visual Grounding

##### Haibo Wang♦, Zihao Lin♦, Zhiyang Xu♢, Lifu Huang♦

♦University of California, Davis ♢Virginia Tech {hibwang, lfuhuang}@ucdavis.edu

#### Abstract

3D Visual Grounding (3D-VG) aims to localize objects in 3D scenes via natural language descriptions. While recent advancements leveraging Vision-Language Models (VLMs) have explored zero-shot possibilities, they typically suffer from a static workflow relying on preprocessed 3D point clouds, essentially degrading grounding into proposal matching. To bypass this reliance, our core motivation is to decouple the task: leveraging 2D VLMs to resolve complex spatial semantics, while relying on deterministic multi-view geometry to instantiate the 3D structure. Driven by this insight, we propose "Think, Act, Build (TAB)", a dynamic agentic framework that reformulates 3D-VG tasks as a generative 2D-to-3D reconstruction paradigm operating directly on raw RGB-D streams. Specifically, guided by a specialized 3D-VG skill, our VLM agent dynamically invokes visual tools to track and reconstruct the target across 2D frames. Crucially, to overcome the multi-view coverage deficit caused by strict VLM semantic tracking, we introduce the Semantic-Anchored Geometric Expansion, a mechanism that first anchors the target in a reference video clip and then leverages multi-view geometry to propagate its spatial location across unobserved frames. This enables the agent to "Build" the target’s 3D representation by aggregating these multi-view features via camera parameters, directly mapping 2D visual cues to 3D coordinates. Furthermore, to ensure rigorous assessment, we identify flaws such as reference ambiguity and category errors in existing benchmarks and manually refine the incorrect queries. Extensive experiments on ScanRefer and Nr3D demonstrate that our framework, relying entirely on open-source models, significantly outperforms previous zero-shot methods and even surpasses fully supervised baselines. Codes will be avaliable at https://github.com/WHB139426/TAB-Agent.

#### 1 Introduction

3D Visual Grounding (3D-VG) (Achlioptas et al., 2020; Chen et al., 2020) is a fundamental task in 3D scene understanding, requiring an AI system to precisely localize a target object within a 3D physical space based on a free-form natural language query. This capability serves as a cornerstone for advanced applications such as human-robot interaction (Kim et al., 2024), embodied AI navigation (Anderson et al., 2018), and AR/VR (Hoenig et al., 2015). Historically, the dominant paradigm in 3D-VG has relied on fully supervised learning frameworks (Roh et al., 2022; Zhao et al., 2021; Zhu et al., 2024; 2025). While these methods achieve remarkable accuracy, their success is predicated on massive amounts of high-quality, densely annotated 3D vision-language datasets. The prohibitive cost and labor-intensive nature of collecting such 3D annotations inherently limit the scalability of supervised methods and their generalization to open-world, open-vocabulary scenarios.

To circumvent the scarcity of 3D annotations, recent efforts have shifted towards zero-shot 3D-VG paradigms (Yang et al., 2023; Yuan et al., 2024b; Zhang et al., 2024; Xu et al., 2025b; Mi et al., 2025). By harnessing pre-trained LLMs (Touvron et al., 2023; Qwen Team, 2026) and VLMs (Bai et al., 2025; Li et al., 2024; Deitke et al., 2025), these methods can ground objects without scene-specific 3D training. However, existing frameworks encounter critical

###### INPUT PROCESS OUTPUT

|Query|
|---|

|Object Table|
|---|

[Figure 1]

- (-) obj 1: laptop, [x, y, z, w, h, l]
- (-) obj 2: couch, [x, y, z, w, h, l]
- (-) obj 3: …

I think obj 2 best matches the query.

Preprocessed 3D Scene Point Clouds

|RGB Video|
|---|

|3D Renderings|
|---|

###### 3D Detector

Target ID

,

|…|
|---|

VLM

Object Table

[Figure 2]

[Figure 3]

Skills

[Figure 4]

[Figure 5]

Act

[Figure 6]

[Figure 7]

[Figure 8]

RGB Video Streams

Reconstructed Object

[Figure 9]

[Figure 10]

[Figure 11]

Tools

Geometric Projection

Build

Think

: Centroid

- Figure 1: (1) Top: Previous methods rely on preprocessed 3D point clouds, degrading the task into proposal matching. (2) Bottom: Our TAB operates directly on RGB-D streams. Through an iterative Think-Act-Build process, the agent reconstructs the target object.

bottlenecks that hinder their real-world deployment. First, the majority of current zero-shot methods (e.g., SeeGround (Li et al., 2025), SPAZER (Jin et al., 2025), SeqVLM (Lin et al., 2025)) heavily rely on well-preprocessed 3D point clouds as inputs. By utilizing static 3D maps to pre-extract candidate bounding boxes, they degrade 3D-VG into a mere “proposal matching” classification task that restricts the VLM to simply selecting from a pre-defined pool of 3D proposals, rendering them ineffective in environments where preprocessed 3D point clouds are unavailable. Second, while some methods attempt to operate directly on 2D images (e.g., VLM-Grounder (Xu et al., 2025b)), they rely exclusively on heuristic 2D semantic matching to associate multi-view observations. By failing to exploit the deterministic 3D geometry inherent in continuous video streams, their tracking becomes highly brittle under extreme viewpoint variations, inevitably yielding fragmented and inaccurate 3D geometries.

To overcome these limitations, we propose Think, Act, Build (TAB), an agentic framework that reformulates zero-shot 3D-VG task from a static proposal matching process into a semantic reasoning and geometric reconstruction process without relying on preprocessed point clouds. Our core motivation stems from the insight that 2D VLMs excel at complex spatial reasoning, leaving the precise 3D structural instantiation entirely to deterministic multi-view geometry. As shown in Figure 1, by directly operating on sequential RGB-D streams, TAB bridges semantic intent with physical space by leveraging VLMs to ground the target into fine-grained 2D masks and then geometrically reconstructing these masks into a 3D object. Rather than adhering to a rigid pipeline, the framework orchestrates a dynamic, iterative process guided by an expert 3D-VG Skill. Following a ReAct-style paradigm (Yao et al., 2022), the agent iteratively “Thinks” by reasoning over the 3D-VG Skill blueprint and current visual context to plan the next step, and “Acts” by invoking specialized tools (e.g.,

- 2D detectors and segmenters) to interact with the visual environment to yield necessary observations for 3D-VG. Crucially, to construct the physical 3D geometry from these 2D observations, the “Build” phase is seamlessly interleaved within this active loop. At its core lies a novel Semantic-Anchored Geometric Expansion mechanism, explicitly designed to overcome the coverage deficits inherent to brittle VLM tracking. By mathematically projecting a semantically anchored 3D centroid across the entire video sequence, this mechanism robustly acquires complete multi-view 2D masks. The agent then utilizes continuous geometric tool invocations to lift these masks directly into 3D space, actively constructing a structurally complete point cloud and estimating a precise 3D bounding box. In summary, our main contributions are as follows:

- • We propose TAB, a novel agentic framework that reformulates zero-shot 3D visual grounding as an active semantic reasoning and geometric reconstruction process.
- • We introduce a Semantic-Anchored Geometric Expansion mechanism to overcome the multi-view coverage deficit inherent in purely semantic-driven tracking.
- • We identify and correct critical flaws such as reference ambiguity and category errors in existing benchmarks (e.g., ScanRefer (Chen et al., 2020) and Nr3D (Achlioptas et al., 2020)) to ensure rigorous assessment for future research.
- • Extensive experiments demonstrate that TAB, powered entirely by open-source models, significantly outperforms state-of-the-art zero-shot methods and even surpasses fully supervised baselines in accurately localizing 3D targets.

- 2 Related Works
- 3D Visual Grounding aims to precisely localize a target object within a 3D scene based on a natural language query. Historically, the field has been dominated by fully supervised methods, primarily divided into two-stage pipelines (Chen et al., 2022; Jain et al., 2022) that rely on pre-trained 3D detectors to generate proposals, and single-stage architectures (Qian et al., 2024; Wu et al., 2023) that directly fuse point cloud and textual features. While achieving strong performance, these approaches are bottlenecked by the prohibitive cost of dense 3D annotations and their limited generalization to open-vocabulary, real-world scenarios. To circumvent this, recent pioneering zero-shot methods, such as LLM-Grounder (Yang et al., 2023), SPAZER (Jin et al., 2025), and SeeGround (Li et al., 2025), have leveraged the remarkable reasoning capabilities of LLMs and VLMs. However, these training-free methods typically employ static workflows and heavily depend on pre-scanned 3D point clouds, essentially degrading the grounding process into a discrete classification task of existing proposals. Furthermore, methods (Xu et al., 2025b) attempting to ground objects from 2D views are often constrained by text-driven semantic matching, making them highly vulnerable to occlusions and extreme camera angles where semantic features degrade. In contrast, our TAB framework reformulates zero-shot 3D-VG as a dynamic agentic process on raw RGB-D videos, leveraging a novel Semantic-Anchored Geometric Expansion to bypass both 3D priors and brittle semantic bottlenecks for robust, generative 3D grounding.

VLMs for 3D Understanding. The remarkable success of 2D Vision-Language Models (VLMs) (Li et al., 2024; Bai et al., 2025; Xu et al., 2025a; Wang et al., 2025a) has spurred extensive efforts to extend their perception capabilities into 3D environments. Current 3D Large Multimodal Models generally construct 3D-aware representations by either employing specialized 3D encoders to process point clouds directly (e.g., PointLLM (Xu et al., 2024), SpatialLM (Mao et al., 2025), VG-LLM (Zheng et al., 2025a)), or by aggregating multi-view

- 2D image features into unified 3D spatial tokens (e.g., 3D-LLM (Hong et al., 2023), LLaVA-
- 3D (Zhu et al., 2025), Video-3D LLM(Zheng et al., 2025b), Ross3D (Wang et al., 2025b)). While these models demonstrate impressive scene-level reasoning and dialogue capabilities, they inherently necessitate resource-intensive cross-modal alignment, relying heavily on massive datasets of paired 3D-text annotations for fine-tuning. Furthermore, their architecture strictly dictates the availability of explicit, dense 3D inputs during inference, such as pre-reconstructed point clouds or voxel grids. In contrast, our TAB framework circumvents the need for 3D-specific pre-training. We harness the inherent reasoning of 2D VLMs within an agentic loop. By orchestrating foundation models with geometric projections, our approach achieves precise 3D spatial understanding directly from raw video streams.

#### 3 Method

Given a language query Q and sequential RGB-D video streams V = {(Ii, Di)}iT=1 consisting of T frames (where Ii is the RGB image and Di is the aligned depth map) with camera intrinsics K and extrinsics Tc2w, TAB directly reconstructs the target object and calculates the 3D bounding box B ∈ R6. As illustrated in Figure 2, TAB formulates 3D-VG as a dynamic agentic loop governed by a comprehensive 3D-VG Skill blueprint. This expert skill serves as the master execution plan, dictating how the VLM agent iteratively engages in a Think (contextual reasoning and planning) and Act (invoking specialized tools) paradigm. Crucially, the Build phase is seamlessly interleaved within this loop to overcome the brittleness of purely semantic tracking. The following sections systematically unpack the core stages of this 3D-VG Skill and the specific visual tools it orchestrates. We detail the tool library in Appendix A.3 and provide a full agent execution trace in Appendix A.4.

##### 3.1 Reference Target Localization

The agent localizes the initial reference target by dynamically orchestrating a suite of semantic tools, guided by its internal reasoning thoughts from the 3D-VG Skill.

Query Analysis. In the initial action, the agent invokes the Query Analysis tool to parse the raw, free-form text Q into a structured JSON format. A complex query like "the pillow on

INPUT PROCESS OUTPUT

[Figure 12]

the pillow on the left bed. it is the top pillow on the side of the bed that is closer to the table between the beds.

Act

(-) Query Analysis (-) Box detect (-) Mask Segment …

Think

Query

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

VLM Agent

Observation

Tool Library

Video

Guide the Process

3D Visual Grounding Skill

###### Semantic-Anchored Geometric Expansion

Semantic expanded video clip with target object

[Figure 17]

3D-VG Skill

Query Query Analysis

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

###### SemanticTemporalExpansion

Fine Filter (Scene Semantics)

Coarse Filter (Object Boxes)

| | |
|---|---|
| | |

[Figure 22]

𝑵𝑵

𝟏𝟏 𝑵𝑵

Geometric Centroid

Initial Build

Frames with target object

𝑷𝑷𝒘𝒘𝒌𝒌

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

𝒌𝒌=𝟏𝟏

intrinsic extrinsic

𝑲𝑲 𝑻𝑻𝒄𝒄𝒄𝒄𝒘𝒘−𝟏𝟏

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

Reference Target

[Figure 31]

Final Build

[Figure 32]

[Figure 33]

[Figure 34]

Seg&&Marker

Score&&Rank

Isolation

Geometric expanded frames with target object

Reference Frame

- Figure 2: The "Think, Act, Build (TAB)" framework. Guided by an expert 3D-VG Skill, the VLM agent reasons about the task, invokes visual tools, and reconstructs the target.

the left bed..." is explicitly disentangled into a target class ("pillow"), visual attributes ("top pillow"), spatial conditions ("closer to the table"), and global scene features ("between the beds"), which are then routed to downstream tools as execution arguments.

Coarse-to-Fine Filtering. To locate candidate frames, the agent executes a two-stage filtering. First, the Coarse Filter tool utilizes foundation detectors to retain frames containing the target class (e.g., identifying frames with beds and pillows). Because conventional detectors fail to distinguish the specific queried instance from same-class distractors, the Fine Filter tool prompts the VLM to rigorously verify if the remaining frames satisfy the parsed scene constraints (e.g., ensuring the frame actually depicts "the table between the beds").

Target Isolation. To transition from frame-level retrieval to instance-level grounding, the agent invokes the Score&Rank tool to evaluate the candidate frames against the query’s parsed attributes and spatial conditions, selecting the most highly-scored image as the Reference Frame. Since multiple instances of the target class may co-exist within a single scene, the agent must resolve intra-class ambiguity. To achieve this, the Seg&Marker tool utilizes foundation segmentation models (e.g., SAM3 (Carion et al., 2026)) to segment all objects of the target class in the reference frame, overlaying a unique numeric ID on each generated mask. During the subsequent Reference Target Isolation step, the VLM actively reasons over this visually prompted image alongside the parsed query to deduce the exact target ID (e.g., singling out the specific top pillow). The agent then isolates and only retains the mask associated with this specific ID. At this stage, the agent has successfully anchored the unambiguous target instance, establishing a pristine visual reference Iref.

##### 3.2 Semantic-Anchored Geometric Expansion

While VLMs exhibit profound semantic reasoning, pure 2D semantic tracking is inherently brittle. As shown by the “pillow” query in Figure 2, textual semantics (e.g., “closer to the table”) frequently fail under extreme viewpoint variations or close-up views lacking context, leading to a multi-view coverage deficit. To overcome this, we introduce the SemanticAnchored Geometric Expansion mechanism, which executes a strategic 2D → 3D → 2D mapping. First, through Semantic Temporal Expansion, the agent tracks the target locally to construct an Initial Build (2D → 3D) and extract a stable 3D geometric centroid (Pcentroid). Next, through Multi-View Geometric Expansion, the agent mathematically projects this Pcentroid onto globally unobserved frames (3D → 2D), leveraging deterministic geometry to acquire complete multi-view masks and robustly bypass VLM semantic blind spots.

Semantic Temporal Expansion. Relying on a single 2D reference frame to lift a 3D centroid is highly susceptible to depth sensor noise and self-occlusion.

As illustrated in the “Semantic Temporal Expansion” arrow of Figure 2, a short-term, multi-frame context is imperative to construct an initial 3D geometry. In Alg 1, given the single reference frame Iref at temporal index tref, the agent exploits the inherent spatiotemporal continuity of the video stream. It initiates a bidirectional tracking loop along the temporal axis. The agent maintains a dynamically growing video context memory Vsem, initialized with the reference frame and its target mask (Iref, Dref, Mref). For each continuously adjacent candidate frame It expanding outwards from tref, the VLM verifies the object’s identity consistency against the context in Vsem. If verified, the foundation segmentation model generates the precise

Algorithm 1: Semantic Temporal Expansion

Input: V = {(Ii, Di)}iT=1, Q, (Iref, Dref, Mref) Output: Vsem

- 1: Vsem ← {(Iref, Dref, Mref)}
- 2: for direction ∆t ∈ {+1, −1} do
- 3: t ← tref + ∆t
- 4: while 1 ≤ t ≤ T do
- 5: if VLM_Verify(Vsem, It, Q) then
- 6: Mt ← Segmentation(It)
- 7: Vsem ← Vsem ∪ {(It, Dt, Mt)}
- 8: t ← t + ∆t
- 9: else
- 10: break
- 11: end if
- 12: end while
- 13: end for

mask Mt, and the comprehensive tuple (It, Dt, Mt) is appended to Vsem to serve as the updated context for the next frame. Crucially, this expansion loop iteratively advances and terminates immediately in a given direction the moment the VLM deduces that the target is no longer present (e.g., due to severe occlusion or moving out of the field of view). This adaptive, semantic-driven tracking effectively captures the object across slightly varying local viewpoints, yielding a continuous and highly reliable semantic video clip

Vsem = {(It, Dt, Mt)}t∈Tlocal, where Tlocal is the set of successfully tracked frame indices.

Centroid Extraction. When tracking frames with the target object, strictly enforcing spatial conditions (e.g., “next to the piano”) severely limits multi-view coverage, as VLMs can mistakenly reject valid frames with close-up or different views of the target object but lacking these contextual backgrounds. To solve this, we abstract the locally tracked 2D pixels into an immutable, viewpoint-invariant 3D physical anchor. Specifically, the agent first reconstructs the target by inverse-projecting only the foreground pixels enclosed by the target mask (i.e., (u, v) ∈ Mt) from each frame It in Vsem into 3D points. Assuming a pinhole camera model, we utilize the true physical depth Dt(u, v) provided by the aligned depth map, alongside the camera intrinsic matrix K ∈ R3×3 to recover the corresponding

- 3D point Pc = [xc, yc, zc]T from the 2D pixel (u, v) in the local camera coordinate system:

 

  (1)

fx 0 cx 0 fy cy 0 0 1

xc yc zc

- u
- v 1

= Dt(u, v) · K−1

, K =

Pc =

where fx, fy represent the focal lengths of the camera, and cx, cy denote the principal point. To aggregate these object points into a globally consistent environment, Pc must also be transformed into the absolute 3D world coordinate system Pw = [xw, yw, zw]T. Using the frame-specific camera extrinsic Tc2w ∈ R4×4, which encapsulates the rotation R ∈ R3×3 and translation t ∈ R3, the transformation is formulated using homogeneous coordinates:

R t 0T 1

##### Pw

Pc

Pc 1

(2)

1 = Tc2w

1 =

Since the inverse-projection is strictly confined to pixels in the masked regions, aggregating these lifted world points Pw across Vsem naturally yields a preliminary 3D point cloud for the target object, denoted as the Initial Build (PCDinit). From this isolated 3D structure, the framework calculates the physical geometric centroid Pcentroid ∈ R3:

N

1 N

Pkw (3)

### ∑

Pcentroid =

k=1

Ambiguous Reference (scene0221_00 )

Object Category Error (scene0702_00 )

Spatial Location Error (scene0658_00)

Raw Query: the cabinet is south of the northwest table. Refined Query: the cabinet is on the left of the table.

Raw Query: the picture on the back wall of the room. Refined Query: the exhaust fan on the back wall of the room.

Raw Query: the bed with a green cover. Refined Query: the bed with a green cover and a laptop.

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

Ambiguous Reference Object Category Error Spatial Location Error

Figure 3: Examples of annotation noise in benchmarks.

255 250 245 245 248 255

243 249 241

where Pkw is the k-th valid point in PCDinit, and N is the total number of valid points. This Pcentroid serves as the spatial anchor for the subsequent multi-view geometric expansion.

Multi-View Geometric Expansion. With the absolute 3D Pcentroid, the agent can geometrically decide whether a given frame Ii contains the target object, bypassing the VLM’s semantic tracking failures. The process follows three key steps: First, the agent mathematically projects the 3D Pcentroid back onto the 2D image plane of frame Ii. By applying the inverse extrinsic matrix T−1

c2w ∈ R4×4 and the intrinsic matrix K, we obtain the theoretical 2D pixel coordinates [u, v]T and the predicted depth zpredict of Pcentroid in frame Ii:

Pc

1 = T−1

c2w

Pcentroid 1

, zpredict

- u
- v 1

= KPc (4)

Second, because this direct projection ignores the Field of View (FoV) boundaries and physical occlusions, the agent performs a strict visibility check on (u, v) and zpredict to ensure the target is actually observable in frame Ii:

(u, v) ∈ Ω

∧ zactual > 0

∧ zpredict ≤ zactual + ϵ Z-buffer Occlusion Check

Depth Validity Check

FoV Boundary Check

(5)

where Ω = [0,W) × [0, H) represents the 2D image domain, zactual = Di(u, v) is the physical depth sampled from the aligned depth map, and ϵ (set to 0.4) accommodates sensor noise

and inherent object thickness. Finally, for each verified frame Ii that passes this check, the 2D coordinate (u, v) serves as a precise point prompt for the segmentation model (e.g.,

SAM3) to extract the target mask Mi. The comprehensive tuple (Ii, Di, Mi) is then appended to the expansion pool: Vgeo ← Vgeo ∪ {(Ii, Di, Mi)}, deterministically acquiring complete multi-view observations ready for the final dense 3D reconstruction.

##### 3.3 2D to 3D Reconstruction

Given the multi-view observations Vgeo, TAB inverse-projects the masked pixels in Vgeo into the 3D world coordinate system following the same process as in Eq. (1) and Eq. (2). To mitigate depth sensor noise and segmentation artifacts, we filter the aggregated raw point cloud using Statistical Outlier Removal (Rusu, 2010) and DBSCAN clustering (Ester et al., 1996) to isolate the main object geometry. Finally, we compute the spatial extremes of this clean cluster to estimate the axis-aligned 3D bounding box B ∈ R6, successfully completing the zero-shot 3D grounding process without relying on pre-scanned point clouds.

Moreover, unlike static zero-shot pipelines that fail upon a single intermediate error, our TAB features robust fault tolerance. By actively monitoring its iterative progress, the agent autonomously recovers from local failures. For example, if a filtering step yields zero candidate images or depth noise corrupts the initial build, it employs a “Dynamic Adjustment” strategy to proactively relax tool thresholds or just skip non-critical steps. This adaptability ensures continuous execution and successful 3D grounding in noisy environments. We also provide an example of fallback execution trace in Appendix A.4.

w/o Unique Multiple Overall PC. Acc@0.25 Acc@0.5 Acc@0.25 Acc@0.5 Acc@0.25 Acc@0.5

Method LLM/VLM

Fully-Supervised Methods (One/Two-Stage Based) ScanRefer (Chen et al., 2020) - ✗ 67.6 46.2 32.1 21.3 39.0 26.1

3DVG-T (Zhao et al., 2021) - ✗ 77.2 58.5 38.4 28.7 45.9 34.5 BUTD-DETR (Jain et al., 2022) - ✗ 84.2 66.3 46.6 35.1 52.2 39.8

EDA (Wu et al., 2023) - ✗ 85.8 68.6 49.1 37.6 54.6 42.3 G3-LQ (Wang et al., 2024) - ✗ 88.6 73.3 50.2 39.7 56.0 44.7

###### Fully-Supervised Methods (LLM/VLM Based)

LLaVA-3D (Zhu et al., 2025) LLaVA-Video-7B ✓ - - - - 50.1 (63.9†) 42.7 (58.6†) Chat-Scene (Huang et al., 2024) Vicuna-7B ✗ 89.5 82.4 47.7 42.9 55.5 50.2

SPAR-mix (Zhang et al., 2025) InternVL-2.5-8B ✓(✗) - - - - 31.9 (48.8) 12.4 (43.1) VG-LLM (Zheng et al., 2025a) Qwen2.5-VL-7B ✓(✗) - - - - 41.6 (57.6) 14.9 (50.9)

Video-3D-LLM (Zheng et al., 2025b) LLaVA-Video-7B ✗ 87.9 78.3 50.9 45.3 58.1 51.7 GPT4Scene (Qi et al., 2026) Qwen2-VL-7B ✗ 90.3 83.7 56.4 50.9 62.6 57.0 3D-RS (Huang et al., 2025) LLaVA-Next-Video ✗ 87.4 77.9 57.0 50.8 62.9 56.1

###### Zero-Shot Methods

LLM-Grounder (Yang et al., 2023) GPT-4 turbo ✗ - - - - 17.1 5.3

ZSVG3D (Yuan et al., 2024b) GPT-4 turbo ✗ 63.8 58.4 27.7 24.6 36.4 32.7 SeeGround (Li et al., 2025) Qwen2-VL-72B ✗ 75.7 68.9 34.0 30.0 44.1 39.4 CSVG (Yuan et al., 2024a) Mistral-Large-2407 ✗ 68.8 61.2 38.4 27.3 49.6 39.8

VLM-Grounder (Xu et al., 2025b) GPT-4o ✓ 66.0 29.8 48.3 33.5 51.6 32.8 SeqVLM (Lin et al., 2025) Doubao-1.5-pro ✗ 77.3 72.7 47.8 41.3 55.6 49.6 SPAZER (Jin et al., 2025) GPT-4o ✗ 80.9 72.3 51.7 43.4 57.2 48.8

TAB (ours) Qwen3-VL-32B ✓(✗) 90.2 (90.2) 57.6 (77.2) 60.1 (60.8) 39.9 (52.5) 71.2 (71.6) 46.4 (61.6)

- Table 1: 3D Visual Grounding results on ScanRefer. "w/o PC" denotes methods that do not rely on 3D point clouds as input. † means results with two stage training.

#### 4 Benchmark Refinement

While benchmarks such as ScanRefer (Chen et al., 2020) and Nr3D (Achlioptas et al., 2020) have significantly advanced 3D-VG, we observe non-negligible annotation noise within the widely adopted evaluation subsets (Xu et al., 2025b). To ensure a rigorous assessment, we manually reviewed and refined these annotations (Figure 3), categorizing the errors into three primary types. First, Ambiguous References lack distinctive features and yield multiple valid candidates; we resolved these by adding exclusive contextual anchors (e.g., appending “and a laptop”). Second, Object Category Errors involve class names contradicting visual reality (e.g., mislabeling an “exhaust fan” as a “picture”), which we corrected to enable accurate semantic anchoring. Finally, Spatial Location Errors describe relationships that contradict the actual 3D layout through erroneous prepositions (e.g., “top left” instead of “bottom”) or invalid global directions (e.g., “south”). We replaced these contradictory coordinates with reliable relative spatial anchors to align with the true 3D geometry.

#### 5 Experiments

##### 5.1 Settings

Benchmarks and Evaluation Metrics. We experiment on the ScanRefer (Chen et al., 2020) and Nr3D (Achlioptas et al., 2020) benchmarks, both built upon ScanNet (Dai et al., 2017) indoor scenes. ScanRefer queries are categorized as “Unique” or “Multiple” depending on the presence of same-class distractors. Its performance is measured by Acc@0.25 and Acc@0.5, representing the fraction of predicted 3D bounding boxes with an IoU > 0.25 and 0.5 against the ground truth. Nr3D queries are divided into “Easy”/“Hard” and “View-Dependent”/“Independent” subsets. Its performance is evaluated by top-1 selection accuracy. Following previous works, our main evaluations are conducted on the widely adopted subsets from recent works (Xu et al., 2025b; Jin et al., 2025; Lin et al., 2025).

Implementation Details. We sample 300 frames per video from the ScanNet image sequences, and build our framework entirely upon open-source models. At its core, we deploy the Qwen3-VL-32B (Bai et al., 2025) as the primary VLM agent. For the foundational vision tools, we utilize Grounding DINO (Liu et al., 2024) for coarse object detection and SAM3 (Carion et al., 2026) for instance segmentation. Both the Semantic Temporal Expansion and the Multi-View Geometric Expansion processes are capped at a maximum of 32 frames.

Method LLM/VLM w/o PC. Easy Hard Dep. Indep. Overall Fully-Supervised Methods

ReferIt3DNet (Achlioptas et al., 2020) - ✗ 43.6 27.9 32.5 37.1 35.6 InstanceRefer (Yuan et al., 2021) - ✗ 46.0 31.8 34.5 41.9 38.8 3DVG-T (Zhao et al., 2021) - ✗ 48.5 34.8 34.8 43.7 40.8

EDA (Wu et al., 2023) - ✗ 58.2 46.1 50.2 53.1 52.1 BUTD-DETR (Jain et al., 2022) - ✗ 54.6 60.7 48.4 46.0 58.0

SceneVerse (Jia et al., 2024) - ✗ 72.5 57.8 56.9 67.9 64.9

###### Zero-Shot Methods

ZSVG3D (Yuan et al., 2024b) GPT-4 turbo ✗ 46.5 31.7 36.8 40.0 39.0 SeeGround (Li et al., 2025) Qwen2-VL-72B ✗ 54.5 38.3 42.3 48.2 46.1

VLM-Grounder (Xu et al., 2025b) GPT-4o ✓ 55.2 39.5 45.8 49.4 48.0 SeqVLM (Lin et al., 2025) Doubao-1.5-pro ✗ 58.1 47.4 51.0 54.5 53.2 SPAZER (Jin et al., 2025) GPT-4o ✗ 68.0 58.8 59.9 66.2 63.8

TAB (ours) Qwen3-VL-32B ✓(✗) 72.1 (72.1) 63.2 (61.4) 62.5 (63.5) 71.4 (69.5) 68.0 (67.2)

- Table 2: 3D Visual Grounding results on Nr3D (w/o GT object class). "w/o PC" denotes methods that do not rely on 3D point clouds as input.

- 5.2 3D Visual Grounding Results ScanRefer. As shown in Table 1, by operating exclusively on raw RGB-D streams without
- 3D point cloud inputs, TAB achieves an overall Acc@0.25 of 71.2% and Acc@0.5 of 46.4%, delivering superior or comparable performance to recent zero-shot and one/two-stage methods. In the challenging "Multiple" subset filled with same-class distractors, TAB achieves 60.1% Acc@0.25. This demonstrates that our framework fully leverages the complex reasoning capabilities of VLMs to analyze fine-grained object conditions and attributes, while our Semantic-Anchored Geometric Expansion effectively overcomes the multi-view coverage deficits inherent to purely semantic 2D tracking. Furthermore, to ensure equitable comparison with proposal-matching approaches, following previous works (Lin et al., 2025; Jin et al., 2025; Li et al., 2025), we also report 3D-assisted results (denoted as "w/o PC." to be "✗") by refining our natively reconstructed bounding box with the Mask3D (Schult et al.,

2023) generated proposal that yields the maximum 3D IoU overlap. Incorporating these proposals yields a substantial surge in Acc@0.5 (46.4% → 61.6%), while Acc@0.25 remains highly stable (71.2% → 71.6%). This dynamic confirms that TAB intrinsically localizes targets with high spatial accuracy; proposal matching merely refines boundary precision against depth map and segmentation noise. Under this 3D-assisted setting, our zero-shot approach significantly surpasses both prior zero-shot and fully-supervised baselines.

Nr3D. As detailed in Table 2, TAB establishes a new zero-shot state-of-the-art with an overall accuracy of 68.0%. It is crucial to note that while conventional methods frequently rely on 3D Scene point clouds provided by the dataset as strong structural priors, TAB strictly operates without any pre-scanned point cloud inputs (denoted as "w/o PC." to be ✓). Despite this strict setting, TAB outperforms prior zero-shot approaches that utilize 3D priors such as SPAZER (63.8%), and surpasses fully-supervised baselines like SceneVerse (64.9%). The framework’s robustness is particularly evident in the challenging "Hard" and "ViewDependent" subsets, achieving accuracies of 63.2% and 62.5%, respectively. These results validate that our framework effectively navigates occlusions and perspective-dependent spatial queries. By actively reasoning over visual semantics to disambiguate complex multiobject references, TAB overcomes the limitations that typically paralyze static pipelines.

##### 5.3 In-Depth Analysis

Single-Frame Reconstruction. We first establish a naive baseline (Table 3 (a)) by disabling both the Semantic Temporal Expansion and the Multi-View Geometric Expansion. In this configuration, the target object is reconstructed only by inverse-projecting the segmentation mask from the single isolated reference frame Iref. As expected, relying on a single 2D view is highly susceptible to depth sensor noise and severe self-occlusion. This lack of multi-view context prevents the agent from observing the object’s full physical extent, resulting in a low overall accuracy of 41.6% Acc@0.25 on ScanRefer and only 52.0% on Nr3D.

Modules ScanRefer Nr3D STE MGE

#

Unique Multiple Overall

Easy Hard Dep. Indep. Overall @0.25 @0.5 @0.25 @0.5 @0.25 @0.5

- (a) ✗ ✗ 57.6 29.3 32.3 19.0 41.6 22.8 58.8 43.8 51.0 52.6 52.0

- (b) ✗ ✓ 65.2 41.3 41.1 22.2 50.0 29.2 62.0 47.1 48.9 59.1 55.1

- (c) ✓ ✗ 69.6 45.7 51.3 30.4 58.0 36.0 67.6 49.1 61.5 57.8 59.2

- (d) ✓ ✓ 90.2 57.6 60.1 39.9 71.2 46.4 72.1 63.2 62.5 71.4 68.0

- Table 3: Ablation study on the Semantic-Anchored Geometric Expansion. STE: Semantic Temporal Expansion. MGE: Multi-View Geometric Expansion.

Effect of Semantic Temporal Expansion. To evaluate the necessity of video context, we bypass the STE phase (Table 3 (b)) and calculate the initial physical centroid directly from the single reference image Iref. This configuration leads to a significant performance drop, particularly in complex scenarios like the Nr3D "Dep." subset (dropping to 48.9%) and ScanRefer’s "Multiple" subset (41.1% Acc@0.25 and 22.2% Acc@0.5). Because a single viewpoint only captures a partial surface, the resulting centroid is heavily biased and spatially offset from the object’s true center of mass. This inaccurate spatial anchor subsequently corrupts the deterministic geometric projection, causing the visibility checks to fetch misaligned 2D views and ultimately derailing the dense reconstruction. This confirms that exploiting spatiotemporal continuity is imperative for establishing a robust initial 3D geometry.

Effect of Multi-View Geometric Expansion. Conversely, we ablate the geometric projection mechanism (Table 3 (c)) by reconstructing the target with only the VLM-tracked frames. While VLMs exhibit profound reasoning capability, tracking driven solely by 2D semantics is inherently brittle. When the camera undergoes extreme viewpoint variations or the target experiences intermediate occlusions, semantic matching frequently fails, causing the temporal expansion to terminate prematurely. This multi-view coverage deficit is reflected in the sharp decline of localization precision: ScanRefer overall Acc@0.5 drops from 46.4% to 36.0%, and performance on the Nr3D "Hard" queries falls from 63.2% to 49.1%.

Qualitative Comparison. Figure 4 provides a visual comparison of our framework against its ablated variants. Our default TAB framework successfully aggregates complete multi-view observations to predict a tight, highly accurate 3D bounding box (IoU = 0.74). In contrast, ablating the Multi-View Geometric Expansion (w/o MGE) restricts the agent to pure semantic tracking, which suffers from view coverage deficits, yielding a fragmented geometry and a reduced IoU of 0.42. Similarly, bypassing the Semantic Temporal Expansion (w/o STE) forces the agent to extract the physical centroid from a single frame. This introduces spatial bias and depth noise into the geometric projection, resulting in an offset and inaccurate bounding box (IoU = 0.35). These visualizations explicitly reinforce the necessity of synergizing both expansion modules for robust 3D localization.

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

Query: a brown ottomon with two black knapsacks and a tissue box on it ... to its left is a black piano

Figure 4: Qualitative comparison of different components in TAB.

#### 6 Conclusion

In this paper, we present Think, Act, Build (TAB), an agentic framework that reformulates zero-shot 3D Visual Grounding into a dynamic reasoning and reconstruction process. By explicitly decoupling semantic understanding from multi-view geometry, TAB orchestrates an iterative “Think” and “Act” loop via 2D Vision-Language Models. To overcome the cov-

erage deficit of purely semantic tracking, our novel Semantic-Anchored Geometric Expansion mechanism projects a 3D centroid across unobserved frames to harvest multi-view masks, seamlessly “Building” a complete 3D point cloud from RGB-D streams. Furthermore, we refine existing 3D-VG benchmarks to establish a rigorous zero-shot evaluation testbed. Ultimately, TAB delivers a highly robust paradigm for 3D scene understanding, demonstrating strong potential for future embodied robotic applications.

#### References

Panos Achlioptas, Ahmed Abdelreheem, Fei Xia, Mohamed Elhoseiny, and Leonidas Guibas. Referit3d: Neural listeners for fine-grained 3d object identification in real-world scenes. In ECCV, pp. 422–440. Springer, 2020.

Peter Anderson, Qi Wu, Damien Teney, Jake Bruce, Mark Johnson, Niko Sünderhauf, Ian Reid, Stephen Gould, and Anton Van Den Hengel. Vision-and-language navigation: Interpreting visually-grounded navigation instructions in real environments. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 3674–3683, 2018.

Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, Wenbin Ge, Zhifang Guo, Qidong Huang, Jie Huang, Fei Huang, Binyuan Hui, Shutong Jiang, Zhaohai Li, Mingsheng Li, Mei Li, Kaixin Li, Zicheng Lin, Junyang Lin, Xuejing Liu, Jiawei Liu, Chenglong Liu, Yang Liu, Dayiheng Liu, Shixuan Liu, Dunjie Lu, Ruilin Luo, Chenxu Lv, Rui Men, Lingchen Meng, Xuancheng Ren, Xingzhang Ren, Sibo Song, Yuchong Sun, Jun Tang, Jianhong Tu, Jianqiang Wan, Peng Wang, Pengfei Wang, Qiuyue Wang, Yuxuan Wang, Tianbao Xie, Yiheng Xu, Haiyang Xu, Jin Xu, Zhibo Yang, Mingkun Yang, Jianxin Yang, An Yang, Bowen Yu, Fei Zhang, Hang Zhang, Xi Zhang, Bo Zheng, Humen Zhong, Jingren Zhou, Fan Zhou, Jing Zhou, Yuanzhi Zhu, and Ke Zhu. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631, 2025.

Nicolas Carion, Laura Gustafson, Yuan-Ting Hu, Shoubhik Debnath, Ronghang Hu, Didac Suris, Chaitanya Ryali, Kalyan Vasudev Alwala, Haitham Khedr, Andrew Huang, et al. Sam 3: Segment anything with concepts. In ICLR, 2026.

Dave Zhenyu Chen, Angel X Chang, and Matthias Nießner. Scanrefer: 3d object localization in rgb-d scans using natural language. In ECCV, pp. 202–221. Springer, 2020.

Dave Zhenyu Chen, Qirui Wu, Matthias Nießner, and Angel X Chang. D 3 net: A unified speaker-listener architecture for 3d dense captioning and visual grounding. In ECCV, pp. 487–505. Springer, 2022.

Angela Dai, Angel X. Chang, Manolis Savva, Maciej Halber, Thomas Funkhouser, and Matthias Nießner. Scannet: Richly-annotated 3d reconstructions of indoor scenes. In CVPR, 2017.

Matt Deitke, Christopher Clark, Sangho Lee, Rohun Tripathi, Yue Yang, Jae Sung Park, Mohammadreza Salehi, Niklas Muennighoff, Kyle Lo, Luca Soldaini, et al. Molmo and pixmo: Open weights and open data for state-of-the-art vision-language models. In CVPR, pp. 91–104, 2025.

Martin Ester, Hans-Peter Kriegel, Jörg Sander, Xiaowei Xu, et al. A density-based algorithm for discovering clusters in large spatial databases with noise. In kdd, volume 96, pp. 226–231, 1996.

Wolfgang Hoenig, Christina Milanes, Lisa Scaria, Thai Phan, Mark Bolas, and Nora Ayanian. Mixed reality for robotics. In IROS, pp. 5382–5387. IEEE, 2015.

Yining Hong, Haoyu Zhen, Peihao Chen, Shuhong Zheng, Yilun Du, Zhenfang Chen, and Chuang Gan. 3d-llm: Injecting the 3d world into large language models. NeurIPS, 2023.

Haifeng Huang, Yilun Chen, Zehan Wang, Rongjie Huang, Runsen Xu, Tai Wang, Luping Liu, Xize Cheng, Yang Zhao, Jiangmiao Pang, et al. Chat-scene: Bridging 3d scene and large language models with object identifiers. NeurIPS, 37:113991–114017, 2024.

Xiaohu Huang, Jingjing Wu, Qunyi Xie, and Kai Han. 3drs: Mllms need 3d-aware representation supervision for scene understanding. In NeurIPS, 2025.

Ayush Jain, Nikolaos Gkanatsios, Ishita Mediratta, and Katerina Fragkiadaki. Bottom up top down detection transformers for language grounding in images and point clouds. In ECCV, pp. 417–433. Springer, 2022.

Baoxiong Jia, Yixin Chen, Huangyue Yu, Yan Wang, Xuesong Niu, Tengyu Liu, Qing Li, and Siyuan Huang. Sceneverse: Scaling 3d vision-language learning for grounded scene understanding. In ECCV, pp. 289–310. Springer, 2024.

Zhao Jin, Rong-Cheng Tu, Jingyi Liao, Wenhao Sun, Xiao Luo, Shunyu Liu, and Dacheng Tao. Spazer: Spatial-semantic progressive reasoning agent for zero-shot 3d visual grounding. In NeurIPS, 2025.

Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan Foster, Grace Lam, Pannag Sanketi, et al. Openvla: An open-source vision-language-action model. arXiv preprint arXiv:2406.09246, 2024.

Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Peiyuan Zhang, Yanwei Li, Ziwei Liu, et al. Llava-onevision: Easy visual task transfer. arXiv:2408.03326, 2024.

Rong Li, Shijie Li, Lingdong Kong, Xulei Yang, and Junwei Liang. Seeground: See and ground for zero-shot open-vocabulary 3d visual grounding. In CVPR, 2025.

Jiawen Lin, Shiran Bian, Yihang Zhu, Wenbin Tan, Yachao Zhang, Yuan Xie, and Yanyun Qu. Seqvlm: Proposal-guided multi-view sequences reasoning via vlm for zero-shot 3d visual grounding. In ACM MM, pp. 3094–3103, 2025.

Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Qing Jiang, Chunyuan Li, Jianwei Yang, Hang Su, et al. Grounding dino: Marrying dino with grounded pre-training for open-set object detection. In ECCV, pp. 38–55. Springer, 2024.

Yongsen Mao, Junhao Zhong, Chuan Fang, Jia Zheng, Rui Tang, Hao Zhu, Ping Tan, and Zihan Zhou. Spatiallm: Training large language models for structured indoor modeling. In NeurIPS, 2025.

Boyu Mi, Hanqing Wang, Tai Wang, Yilun Chen, and Jiangmiao Pang. Language-to-space programming for training-free 3d visual grounding. In EMNLP, pp. 3844–3864, 2025.

Zhangyang Qi, Zhixiong Zhang, Ye Fang, Jiaqi Wang, and Hengshuang Zhao. Gpt4scene: Understand 3d scenes from videos with vision-language models. In ICLR, 2026.

Zhipeng Qian, Yiwei Ma, Zhekai Lin, Jiayi Ji, Xiawu Zheng, Xiaoshuai Sun, and Rongrong Ji. Multi-branch collaborative learning network for 3d visual grounding. In ECCV, pp. 381–398. Springer, 2024.

Qwen Team. Qwen3.5: Towards native multimodal agents, February 2026. URL https: //qwen.ai/blog?id=qwen3.5.

Junha Roh, Karthik Desingh, Ali Farhadi, and Dieter Fox. Languagerefer: Spatial-language model for 3d visual grounding. In CoRL, pp. 1046–1056. PMLR, 2022.

Radu Bogdan Rusu. Semantic 3d object maps for everyday manipulation in human living environments. KI-Künstliche Intelligenz, 24(4):345–348, 2010.

Jonas Schult, Francis Engelmann, Alexander Hermans, Or Litany, Siyu Tang, and Bastian Leibe. Mask3D: Mask Transformer for 3D Semantic Instance Segmentation. In ICRA, 2023.

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv:2302.13971, 2023.

Haibo Wang, Bo Feng, Zhengfeng Lai, Mingze Xu, Shiyu Li, Weifeng Ge, Afshin Dehghan, Meng Cao, and Ping Huang. Streambridge: Turning your offline video large language model into a proactive streaming assistant. In NeurIPS, 2025a.

Haochen Wang, Yucheng Zhao, Tiancai Wang, Haoqiang Fan, Xiangyu Zhang, and Zhaoxiang Zhang. Ross3d: Reconstructive visual instruction tuning with 3d-awareness. In CVPR, pp. 9275–9286, 2025b.

Yuan Wang, Yali Li, and Shengjin Wang. Gˆ 3-lq: Marrying hyperbolic alignment with explicit semantic-geometric modeling for 3d visual grounding. In CVPR, pp. 13917–13926, 2024.

Yanmin Wu, Xinhua Cheng, Renrui Zhang, Zesen Cheng, and Jian Zhang. Eda: Explicit text-decoupling and dense alignment for 3d visual grounding. In CVPR, pp. 19231–19242, 2023.

Mingze Xu, Mingfei Gao, Shiyu Li, Jiasen Lu, Zhe Gan, Zhengfeng Lai, Meng Cao, Kai Kang, Yinfei Yang, and Afshin Dehghan. Slowfast-llava-1.5: A family of token-efficient video large language models for long-form video understanding. In COLM, 2025a.

Runsen Xu, Xiaolong Wang, Tai Wang, Yilun Chen, Jiangmiao Pang, and Dahua Lin. Pointllm: Empowering large language models to understand point clouds. In ECCV, pp. 131–147. Springer, 2024.

Runsen Xu, Zhiwei Huang, Tai Wang, Yilun Chen, Jiangmiao Pang, and Dahua Lin. Vlmgrounder: A vlm agent for zero-shot 3d visual grounding. In CoRL, 2025b.

Jianing Yang, Xuweiyi Chen, Shengyi Qian, Nikhil Madaan, Madhavan Iyengar, David F. Fouhey, and Joyce Chai. Llm-grounder: Open-vocabulary 3d visual grounding with large language model as an agent. In ICRA, pp. 7694–7701, 2023.

Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik R Narasimhan, and Yuan Cao. React: Synergizing reasoning and acting in language models. In ICLR, 2022.

Qihao Yuan, Kailai Li, and Jiaming Zhang. Solving zero-shot 3d visual grounding as constraint satisfaction problems. arXiv preprint arXiv:2411.14594, 2024a.

Zhihao Yuan, Xu Yan, Yinghong Liao, Ruimao Zhang, Sheng Wang, Zhen Li, and Shuguang Cui. Instancerefer: Cooperative holistic understanding for visual grounding on point clouds through instance multi-level contextual referring. In ICCV, pp. 1791–1800, 2021.

Zhihao Yuan, Jinke Ren, Chun-Mei Feng, Hengshuang Zhao, Shuguang Cui, and Zhen Li. Visual programming for zero-shot open-vocabulary 3d visual grounding. In CVPR, 2024b.

Jiahui Zhang, Yurui Chen, Yanpeng Zhou, Yueming Xu, Ze Huang, Jilin Mei, Junhui Chen, Yu-Jie Yuan, Xinyue Cai, Guowei Huang, et al. From flatland to space: Teaching visionlanguage models to perceive and reason in 3d. arXiv preprint arXiv:2503.22976, 2025.

Sha Zhang, Di Huang, Jiajun Deng, Shixiang Tang, Wanli Ouyang, Tong He, and Yanyong Zhang. Agent3d-zero: An agent for zero-shot 3d understanding. In ECCV, pp. 186–202. Springer, 2024.

Lichen Zhao, Daigang Cai, Lu Sheng, and Dong Xu. 3dvg-transformer: Relation modeling for visual grounding on point clouds. In ICCV, pp. 2928–2937, 2021.

Duo Zheng, Shijia Huang, Yanyang Li, and Liwei Wang. Learning from videos for 3d world: Enhancing mllms with 3d vision geometry priors. In NeurIPS, 2025a.

Duo Zheng, Shijia Huang, and Liwei Wang. Video-3d llm: Learning position-aware video representation for 3d scene understanding. In CVPR, pp. 8995–9006, 2025b.

Chenming Zhu, Tai Wang, Wenwei Zhang, Kai Chen, and Xihui Liu. Scanreason: Empowering 3d visual grounding with reasoning capabilities. In ECCV, pp. 151–168. Springer, 2024.

###### Chenming Zhu, Tai Wang, Wenwei Zhang, Jiangmiao Pang, and Xihui Liu. Llava-3d: A simple yet effective pathway to empowering lmms with 3d capabilities. In ICCV, pp. 4295–4305, 2025.

#### A Appendix

##### A.1 Expert Skill: 3D Visual Grounding

The expert skill serves as the blueprint for the agent, defining the standard operating procedure for the 3D visual grounding and reconstruction pipeline. It is structured as a Markdown document designed to be directly read and parsed by the agent.

Skill: 3D Visual Grounding Pipeline

Name: 3d_visual_grounding Description: Expert Skill for 3D Visual Grounding & Reconstruction Pipeline. Use this when the user gives you a scene_id and a query.

When to use this skill: Use this skill when the user requests to find and reconstruct a specific object in a scene (video with ∼300 frames) and calculate its 3D bounding box coordinates.

Instructions: You MUST strictly follow this sequential execution path:

- 1. Query Analysis: Parse the user’s natural language query to extract the target_class, visual attributes, spatial conditions, and global scene_features.
- 2. Initialize Image Registry: Call the read tool to index all image file names from the scene video into a standard JSON list.
- 3. Coarse Filtering (Object Masks): Apply mask-based filtering to retain only images containing the target_class. Critical Check: If 0 images remain, do not proceed. Analyze if the threshold is too high and retry with a lower value.
- 4. Fine Filtering (Scene Semantics): Apply VLM-based filtering to verify if images match the scene_feature. Fallback Strategy: If this filter removes ALL images, discard this step and revert to the result of Coarse Filtering (Object Masks). Use the images filtered by Coarse Filtering (Object Masks) as the images.
- 5. Scoring & Ranking: Score the remaining images based on their alignment with the query’s attributes and conditions. Sort in descending order.
- 6. Optimal View & Target Selection: Traverses the candidate images to find the most distinct observation of the scene as the “Reference View”, and uses VLM to identify the specific Object ID within this view that best matches the query description.
- 7. Reference Target Isolation: Generate a visualization isolating the specific Object ID.
- 8. Temporal Expansion (Video Tracking): Expand the target object search temporally from the Reference View. It tracks the object frame-by-frame (forward and backward) using the VLM to verify identity and SAM to generate masks. This ensures temporal consistency and generates a continuous video clip with the target object. This leverages temporal consistency to robustly collect a continuous video clip containing the target as candidate frames.
- 9. Candidate Segmentation: For every verified candidate frame, generate and save the precise segmentation mask for the target object.
- 10. Initial 3D Reconstruction: Aggregate all verified images and their masks from the temporal expansion steps to generate an initial, preliminary 3D point cloud.
- 11. Geometric Multi-View Expansion (Centroid Complete): Extract the target’s absolute 3D centroid and mathematically project it across other scene frames, utilizing camera extrinsics and Z-Buffer checks.
- 12. Final Dense 3D Reconstruction: Reconstruct the updated, comprehensive image and mask paths into a final, dense, and structurally complete PLY point cloud.
- 13. 3D Bounding Box Calculation: Calculate the axis-aligned 3D bounding box.

Strategy Tips: Dynamic Adjustment: If a specific step returns 0 valid images, DO NOT give up immediately. Consider relaxing constraints or skipping it. Threshold Consistency: If you lower the threshold in an earlier step, you MUST maintain this lowered threshold for ALL subsequent steps.

##### A.2 Prompts

We provide the prompts utilized across different modules of our framework. These prompts dictate the reasoning, filtering, and tracking behaviors of the VLM agent.

Agent System Prompt

You are an intelligent assistant capable of calling external tools. ### YOUR EXPERT SKILLS {skill_description} ### AVAILABLE TOOLS The available tools and their STRICT parameter definitions are as follows: {tool_descriptions} ### RESPONSE FORMAT Please respond in a step-by-step loop. For each step, output a Thought followed by an Action, strictly according to the following format: Thought:

- • First, recall "YOUR EXPERT SKILLS" to decide which skill to use and the current pipeline stage.
- • Analyze the previous Observation.
- • Plan the next step accordingly.
- • CRITICAL: Check if the previous observation indicates failure. If so, consider retrying with looser constraints or ABORTING.

Action: The action you decide to take. It MUST be formatted EXACTLY as one of the following:

- 1. Tool Call: <tool_name>(<tool_input_json>)

- • <tool_name> must be one of: [{tool_names}]
- • <tool_input_json> must be a valid JSON string containing arguments matching the tool’s schema.
- • Example: filter_by_mask({"threshold": 0.5})

- 2. Finish Task: Finish[<your final answer here>]

• Use this when you have obtained the final answer or completed the task successfully.

- 3. Abort Task: Abort[<failure_reason>]

• Use this when the task CANNOT be completed.

###### Query Parsing Prompt

You are working on a 3D visual grounding task. You will receive a natural language query that specifies a particular object by describing its attributes and grounding conditions in a scene. Definitions:

- • Target object phrase: The core noun phrase identifying the object. Unlike a simple class name, this must include 1-2 key adjectives from the query if present (e.g., use "rectangular dark cabinet" instead of just "cabinet"). Include inherent adjectives (color, shape, material). Do NOT include spatial or relative adjectives.
- • Attributes: Inherent properties of the target object itself, such as category description, color, material, shape, appearance, function, or state.
- • Grounding conditions: Relational, spatial, or contextual constraints that help uniquely locate the target object relative to other objects, regions, or layouts in the scene.
- • Scene feature: A single sentence describing the scene’s composition based STRICTLY on the objects, regions, and layouts explicitly mentioned in the query. Do NOT infer or hallucinate context.

Your task: 1. Parse the query. 2. Identify and return: the target object’s phrase, a list of the object’s attributes, a list of grounding conditions, and a single string describing the scene feature. Your response must be formatted strictly as JSON wrapped inside triple backticks:

{

"target_class": "...", "attributes": [...], "conditions": [...], "scene_feature": "..."

}

###### Scene Filter Prompt

You are a strict visual verification assistant. Your task is to analyze an image and determine if it matches the scene description provided by the user. You must rigorously verify the image against the user’s text across these specific dimensions:

- 1. Objects: Confirm that EVERY physical object mentioned in the description is clearly visible. IMPORTANT EXCEPTION: Ignore broad functional area labels or room types (e.g., "living room", "kitchen"). Focus ONLY on specific physical items.
- 2. Quantity: Strictly count the objects. If the text specifies a number, the image must show equal to or more than that count.
- 3. Position & Space: Verify the spatial layout, but RELAX directional strictness:

- • Ignore "Left/Right": Due to variable camera angles, strictly ignore absolute instructions like "to the left of".
- • Adjacency Rule: Treat any mention of "left" or "right" simply as "next to" or "nearby".
- • Other Spatial Relations: Maintain strict verification for non-directional spatial terms (e.g., "on top of", "under").

- 4. Attributes: Check that key visual attributes (e.g., color, material) match the description. CRITICAL INSTRUCTION: This is a strict verification task, EXCEPT for the left/right relaxation mentioned above. Only answer "yes" if all constraints are satisfied. Do not provide explanations, just directly answer "yes" or "no".

###### VLM Score Prompt

You are a visual filtering assistant. Your goal is to determine if the Target Object described in the user’s query is visible in the provided image. ### Evaluation Steps

- 1. Target Detection (The Strict Gatekeeper): Focus ONLY on the target_class. Is this specific object visible in the image? If NO: The score MUST be exactly 0.0.
- 2. Context Verification: Check the conditions regarding other objects (Reference Objects). Are the reference objects mentioned visible? If MISSING: The score must be low (0.1 - 1.9).
- 3. Attribute Matching: Evaluate visual details (color, shape, state). ### Confidence Score Scale (Float)

- • 0.0: Target Absent.
- • 0.1 - 1.9: Present, but Context Missing / Strong Mismatch.
- • 2.0 - 2.9: Present, Low Match.
- • 3.0 - 3.9: Present, Partial Match / Ambiguous.
- • 4.0 - 4.9: Present, High Match.
- • 5.0: Present, Perfect Match. Output Format: You must return a strict JSON object containing is_present (Boolean) and score (Float). Start directly with { and end with }.

###### Frame Expansion Prompt

You are an expert in Video Object Tracking and Visual Re-Identification. ### TASK Your task is to determine if a specific Target Object (highlighted by bounding boxes in the Reference Video) is present in the provided Candidate Image.

### CONTEXT The Candidate Image is an adjacent frame (temporally continuous) to the Reference Video clip. The object, if present, will have a similar appearance, scale, and location, subject to minor motion or camera movement.

###### ### JUDGMENT CRITERIA

- 1. Identity Consistency: The object must be the exact same instance, not just a similar category.
- 2. Robustness: Allow for slight changes in viewpoint, lighting, scale, or partial occlusion due to motion.
- 3. Strict Rejection: If the object is fully occluded or has moved out of the frame, answer NO. Output strictly one word. If the target is present: "YES". If the target is absent: "NO".

###### Segmentation Marker Prompt

You are an expert who identifies objects in annotated scenes. You will receive an Annotated Image with bounding boxes and numeric IDs (0, 1, 2, ...) highlighting the objects. ### CRITICAL VISUAL GUIDELINES

- 1. Focus Inside the Box: The bounding box delimits the area of interest. To verify attributes, you must analyze the visual content strictly within the boundaries of the box for that ID.
- 2. Spatial Reasoning (High Priority): Pay strict attention to spatial descriptors in the query (e.g., "leftmost," "behind," "next to").
- 3. Object Matching: Ensure the object inside the bounding box matches the target_class. ### Your Task

- • Single Candidate Override: If there is ONLY ONE object annotated (only ID 0 exists), strictly output ID: 0 immediately.
- • Strict Verification (Multi-object): Scan the annotated image to find the object ID that matches the class, spatial conditions, and attributes.
- • Failure Handling: If NONE satisfy all constraints, output ID: -1. Output ONLY the ID in the format: ID: <number>

##### A.3 Tool Library

The TAB agent is equipped with a comprehensive library of specialized tools. These tools encapsulate foundation vision models (e.g., SAM), Vision-Language Models, and multi-view geometric projection functions. The agent dynamically invokes these tools via strict JSON parameter schemas to execute its planned actions. Table 4 details the complete tool registry.

Tool Name Description query_parse() Parses the natural language query into a structured

JSON format containing target_class, attributes, conditions, and scene_feature.

read_image_files() Scans the local directory for a specific scene and indexes all image file paths into a structured list.

object_filter() Filters candidate images using GroundingDINO to retain only frames containing the requested target_class.

vlm_filter() Utilizes a Vision-Language Model to verify if an image strictly satisfies the global scene constraints.

vlm_score() Scores and ranks candidate images based on how well their visual contents match the query’s attributes and spatial conditions.

argmax_image_and_seg_id() Select the best candidate image and identify the specific target object ID within it. It iterates through the images (from highest score), generates segmentation masks using the SAM, and uses the VLM to pinpoint the specific object ID matching the query.

segment_target_in_reference() Isolate the specific target object in the reference image. It draws a clean bounding box around the identified target ID to create a ’Reference View’.

vlm_frame_expansion() Expand the target object search temporally from a reference frame. It tracks the object frame-by-frame (forward and backward) using the VLM to verify identity and the SAM to generate masks.

segment_all_target_object() Perform segmentation on all candidate images for reconstruction. It iterates through the list of validated images, generates segmentation masks, and uses the VLM to identify and save the specific mask corresponding to the target object in each view.

reconstruct_point_cloud() Generates the 3D point cloud by lifting the segmented target images back into 3D space using camera parameters.

centroid_complete() Extracts the target’s 3D centroid and mathematically projects it across unobserved frames with depth-based occlusion checks, maximizing view coverage.

calculate_bbox() Calculates and outputs the final axis-aligned 3D bounding box from the reconstructed target point cloud.

Table 4: The complete registry of specialized tools available to the agent.

##### A.4 Agent Execution Trace Example

To concretely demonstrate how the framework operates in practice, we provide a complete execution trace for a complex query. The agent successfully follows the 3D Visual Grounding skill pipeline, from initial query parsing to the final 3D bounding box calculation.

Scene ID: scene0435_00 Query: “the pillow on the left bed. it is the top pillow on the side of the bed that is closer to the table between the beds.”

Execution Trace

- [Step 1] Query Analysis Thought:

- • I need to begin the 3D Visual Grounding & Reconstruction Pipeline as per the instructions.
- • The first step is to parse the user’s query to extract structured information: target_class, attributes, conditions, and scene_feature.
- • This will guide all subsequent steps in the pipeline. Action: query_parse({"query": "..."}) Observation:

{

"target_class": "pillow", "attributes": ["top pillow"], "conditions": ["it's on the left bed", "it's the top pillow",

"it's on the side... closer to the table..."],

"scene_feature": "The scene contains two beds with a table between them..." }

- [Step 2] Initialize Image Registry Thought:

- • The query has been successfully parsed into structured JSON, extracting the target class (pillow), conditions, and scene feature.
- • The next step is to initialize the image registry by reading all image file paths from the scene directory.
- • This will provide the full set of frames to work with for subsequent filtering and reconstruction

steps. Action: read_image_files({"scene_id": "scene0435_00"}) Observation: image_files.json

- [Step 3] Coarse Filtering Thought:

- • The image registry has been initialized successfully with all image paths stored in ’image_files.json’.
- • The next step is to perform Coarse Filtering using mask-based detection to retain only images containing the target class (’pillow’).
- • I will use the default threshold of 0.5 for mask generation. If no images are found, I will retry with a lower threshold as per the failure strategy.

Action: object_filter({"threshold": 0.5, ...}) Observation: object_filtered_image_files.json, 90 (Images retained)

- [Step 4] Fine Filtering Thought:

- • Coarse Filtering successfully returned 90 images containing the target class ’pillow’.
- • The next step is Fine Filtering using VLM to verify if these images match the scene_feature: "The scene contains two beds with a table between them, and the left bed has a pillow on the side closer to the table."
- • This step will help narrow down images that match the broader scene context. Action: vlm_filter({...}) Observation: vlm_filtered_image_files.json, 24 (Images retained)

- [Step 5] Scoring & Ranking Thought:

- • VLM-based Fine Filtering returned 24 images that match the scene feature.

- • The next step is to score and rank these images based on their alignment with the full query (including attributes and conditions).
- • This will help prioritize the most relevant views for subsequent steps. Action: vlm_score({...}) Observation: (’vlm_scores.json’, ’vlm_ranked_image_files.json’)

###### [Step 6] Optimal View & Target Selection Thought:

- • The images have been successfully scored and ranked based on relevance to the query.
- • The next step is to select the best candidate image and identify the specific target object ID within it.
- • I will use the highest-scoring image and apply the same threshold (0.5) used before for consistency.

Action: argmax_image_and_seg_id({...}) Observation: (’.../00085.jpg’, 5, ’reference_image_mask_results.pth’)

###### [Step 7] Reference Target Isolation Thought:

- • The best candidate image has been identified: ’.../00085.jpg’.
- • The specific target object ID is 5.
- • The next step is to isolate the target object in this reference image by generating a clean visualization with a bounding box.

Action: segment_target_in_reference({...}) Observation: reference_image_with_target_mask_box.png

###### [Step 8] Temporal Expansion Thought:

- • The reference image with the target pillow isolated and masks has been successfully generated.
- • The next step is to perform temporal expansion from this reference frame to track the target object across frames.
- • I will use a maximum of 32 frames for expansion and maintain the threshold of 0.5 for consistency.

Action: vlm_frame_expansion({...}) Observation: (’expanded_image_files.json’, 10)

###### [Step 9] Candidate Segmentation Thought:

- • Temporal expansion from the reference frame yielded 10 images for 3D reconstruction.
- • I will proceed to segment the target object in all candidate images to prepare for reconstruction.
- • This step will generate masks for each image, which are necessary for the reconstruction. Action: segment_all_target_object({...}) Observation: (’final_images.json’, ’final_masks.pth’)

###### [Step 10] Initial 3D Reconstruction Thought:

- • The target object has been segmented in all candidate images, producing final image and mask files.
- • The next step is to perform initial 3D reconstruction using these images and masks.
- • This will create a preliminary point cloud for the target pillow. Action: reconstruct_point_cloud({...}) Observation: pred_pcd.ply

###### [Step 11] Geometric Multi-View Expansion Thought:

- • The initial 3D point cloud has been successfully reconstructed and saved as ’pred_pcd.ply’.
- • The next critical step is to perform Geometric Multi-View Expansion using the centroid_complete tool to maximize view coverage.
- • This will project the 3D centroid across other scene frames and harvest additional unoccluded views for a more complete reconstruction.

Action: centroid_complete({...}) Observation: (’centroid_final_images.json’, ’centroid_final_masks.pth’)

###### [Step 12] Final Reconstruction Thought:

- • The centroid_complete tool has successfully expanded the view coverage, producing a more comprehensive set of images and masks.
- • The next step is to perform the final dense 3D reconstruction using these updated inputs.
- • This will generate a denser and more structurally complete point cloud. Action: reconstruct_point_cloud({...}) Observation: pred_pcd.ply

###### [Step 13] 3D Bounding Box Calculation Thought:

- • The final dense 3D point cloud has been successfully reconstructed and saved as ’pred_pcd.ply’.
- • The next and final step is to calculate the axis-aligned 3D bounding box of this point cloud.
- • This will provide the required coordinates for the target pillow. Action: calculate_bbox({"pcd_path": "pred_pcd.ply"}) Observation: [1.6515, 1.1065, 0.7770, 0.4687, 0.6466, 0.2580]

[Final Output] Thought:

- • The 3D bounding box has been successfully calculated for the point cloud of the target pillow.
- • The bounding box coordinates are: [cx cy cz dx dy dz] = [1.6515, 1.1065, 0.7770, 0.4687, 0.6466, 0.2580].
- • This completes the entire 3D Visual Grounding & Reconstruction Pipeline. Action: Finish[The 3D bounding box for the pillow on the left bed is [1.6515, 1.1065, 0.7770, 0.4687, 0.6466, 0.2580].]

While the complete trace above illustrates a perfectly smooth execution, it is important to note that our framework is designed with strong robustness and dynamic fallback mechanisms to handle challenging or ambiguous observations. If the agent encounters an error or invalid state during an intermediate step, it does not simply crash; instead, it gracefully falls back to a previous stable state. For instance, if the Geometric Multi-View Expansion step calculates an invalid NaN centroid due to depth sensor noise or errors, the framework will automatically abort this specific expansion and proceed to perform the 3D reconstruction directly using the valid frames already gathered during the Temporal Expansion phase.

Furthermore, the agent actively self-corrects during semantic and visual filtering through Dynamic Adjustment and Threshold Consistency strategies. If a strict initial constraint (e.g., a default segmentation threshold of 0.5) results in zero valid candidates, the agent does not immediately abort the task. Instead, it dynamically relaxes the parameter to recover the target, and strictly maintains this updated threshold across all subsequent tracking and segmentation modules to prevent logical contradictions. To demonstrate this intelligent self-correction, the following trace snippet showcases the agent recovering from a failed Coarse Filtering step and preserving the modified context:

Fallback Execution Trace

[... Initial Steps Skipped ...] [Step 3] Coarse Filtering (Object Masks) Thought:

- • The image registry has been initialized successfully with all image paths stored in ’image_files.json’.
- • The next step is to perform Coarse Filtering using object-based detection to retain only images containing the target class (’small black remote’).
- • I will use the default threshold of 0.5. Action: object_filter({"image_files_path": "image_files.json", "parsed_query": "...", "threshold": 0.5}) Observation: object_filtered_image_files.json, 0

###### [Step 3] Coarse Filtering (Object Masks) - Retry Thought:

- • CRITICAL: The previous filtering step returned 0 valid images. I MUST NOT give up immediately.
- • Following the Dynamic Adjustment strategy, the threshold of 0.5 might be too strict for this specific small object. I will retry with a looser threshold of 0.3.
- • I must also remember to maintain this 0.3 threshold for all subsequent steps according to the Threshold Consistency rule.

Action: object_filter({"image_files_path": "image_files.json", "parsed_query": "...", "threshold": 0.3}) Observation: object_filtered_image_files.json, 24

###### [... Intermediate Steps Skipped ...] [Step 6] Optimal View & Target Selection Thought:

- • The images have been successfully scored and ranked.
- • I will select the best candidate image and identify the specific target object ID within it.
- • CRITICAL: Following the Threshold Consistency rule, I MUST use the lowered threshold of 0.3 that was established during Coarse Filtering, rather than reverting to the default 0.5.

Action: argmax_image_and_seg_id({"scores_path": "...", "image_files_path": "...", "query": "...", "parsed_query": "...", "threshold": 0.3}) Observation: (’.../00112.jpg’, 2, ’reference_image_mask_results.pth’)

