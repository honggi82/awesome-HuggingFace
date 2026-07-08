# arXiv:2508.17198v1[cs.AI]24Aug2025

## From reactive to cognitive: brain-inspired spatial intelligence for embodied agents

##### Shouwei Ruan1,2†, Liyuan Wang1,3†, Caixin Kang2, Qihui Zhu2, Songming Liu1, Xingxing Wei2* and Hang Su1*

1Department of Computer Science and Technology, Institute for AI, BNRist Center, Tsinghua-Bosch Joint ML Center, THBI Lab, Tsinghua University, Beijing, China. 2Institute of Artificial Intelligence, Beihang University, Beijing, China. 3Department of Psychological and Cognitive Sciences, Tsinghua University, Beijing, China.

*Corresponding author(s). E-mail(s): xxwei@buaa.edu.cn; suhangss@mail.tsinghua.edu.cn; Contributing authors: shouweiruan@buaa.edu.cn; liyuanwang@tsinghua.edu.cn; caixinkang@buaa.edu.cn; sy2202227@buaa.edu.cn; lsm23@mails.tsinghua.edu.cn; †These authors contributed equally to this work.

Abstract Spatial cognition enables adaptive goal-directed behavior by constructing internal models of space. Robust biological systems consolidate spatial knowledge into three interconnected forms: landmarks for salient cues, route knowledge for movement trajectories, and survey knowledge for map-like representations. While recent advances in multi-modal large language models (MLLMs) have enabled visual-language reasoning in embodied agents, these efforts lack structured spatial memory and instead operate reactively, limiting their generalization and adaptability in complex real-world environments. Here we present Brain-inspired Spatial Cognition for Navigation (BSC-Nav), a unified framework for constructing and leveraging structured spatial memory in embodied agents. BSC-Nav builds allocentric cognitive maps from egocentric trajectories and contextual cues, and dynamically retrieves spatial knowledge aligned with semantic goals. Integrated with powerful MLLMs, BSC-Nav

1

achieves state-of-the-art efficacy and efficiency across diverse navigation tasks, demonstrates strong zero-shot generalization, and supports versatile embodied behaviors in the real physical world, offering a scalable and biologically grounded path toward general-purpose spatial intelligence. Our code is available at https://github.com/Heathcliff-saku/ BSC-Nav, and the Supplementary video is available at Google Drive

Keywords: spatial navigation, spatial cognition, cognitive map, neuro-inspired learning, embodied intelligence

### 1 Introduction

Spatial cognition, the ability to acquire, organize, exploit, and update knowledge about external space, is fundamental to both human beings and artificial intelligence (AI) [1, 2]. It underlies not only sensorimotor skills such as navigation and manipulation, but also supports higher-level cognitive functions, including abstraction, planning, and reasoning. In humans, generalizable spatial representations support interpretation of sensory input, anticipation of future events, and flexible adaptation to changing environments, from brewing coffee in a familiar kitchen to navigating an unfamiliar city [3–5]. Given its broad relevance in embodied interaction with the real physical world, spatial cognition has emerged as a central theme across disciplines, driving advances in robotics [6], urban simulation [7], and planetary-scale modeling [8], while increasingly recognized as a foundational component of artificial general intelligence (AGI) [3, 9, 10].

Despite the rapid progress in multi-modal large language models (MLLMs) and growing efforts to equip embodied agents with visual-language reasoning [11–13], current artificial systems remain fundamentally limited in large-scale spatial cognition [14], especially in tasks requiring long-horizon navigation and mobile manipulation. A central bottleneck lies in the lack of structured spatial memory [15–17], a mechanism for persistently encoding, organizing, and retrieving spatial knowledge about the environments. Most existing methods, whether based on end-to-end reinforcement learning [18–20] or modular pipelines with powerful MLLMs [21, 22], process observations in a reactive and stateless manner. Without durable internal models of external space, agents struggle to consolidate coherent spatial representations or perform reasoning beyond immediate stimuli, resulting in fragmented knowledge, short-sighted planning, and poor generalization. Addressing these limitations requires a paradigm shift from reactive processing to memory-centric spatial cognition, which supports persistent representations and compositional reasoning over time and space.

Compared to artificial systems, biological spatial cognition offers a robust and compelling template. Decades of neuroscience research revealed that organisms consolidate spatial knowledge into three distinct yet interconnected

forms [16, 17] (Fig. 1a): landmarks, which encode stable associations of salient environmental cues to support localization and contextual understanding [23, 24]; route knowledge, which captures egocentric movement trajectories between landmarks for habitual navigation and path integration [25]; and survey knowledge, which integrates multiple routes into allocentric, map-like representations that support flexible inference, shortcut discovery, and detour planning [26]. These spatial representations are accessed and coordinated via working memory [27], especially visual-spatial working memory [28, 29], enabling adaptive retrieval, composition, and generalization based on task demands and environmental familiarity.

Motivated by these biological principles, we introduce Brain-inspired Spatial Cognition for Navigation (BSC-Nav), a unified framework that instantiates cognitive spatial intelligence in embodied agents via structured spatial memory (Fig. 1b). BSC-Nav explicitly constructs spatial knowledge with two synergistic branches: a landmark memory module, which encodes durable associations between salient environmental cues and spatial locations to instantiate landmarks; and a cognitive map module, which accumulates route knowledge by transforming egocentric movement sequences into voxelized trajectories, and organizes them into allocentric, map-like representations as survey knowledge. BSC-Nav further incorporates a working memory module that dynamically retrieves and combines spatial representations from landmark memory and cognitive map, thereby aligning semantic goals with grounded spatial actions. Each component interfaces seamlessly with large-scale foundation models: visual foundation models like DINOv2 [30, 31] provide perceptual grounding of environmental cues, while MLLMs like GPT-4V [21] guide high-level semantic interpretation and goal-conditioned reasoning.

By integrating MLLMs-based embodied agents with structured spatial memory, BSC-Nav achieves robust spatial cognition capabilities, supporting long-horizon reasoning, experience reuse, and flexible transitions between local and global policies. BSC-Nav achieves state-of-the-art performance across a broad spectrum of navigation tasks, including object-goal, open-vocabulary, and instance-level navigation, while also demonstrating strong spatial generalization in instruction following, embodied question answering, and mobile manipulation. These results position BSC-Nav as a scalable and biologically grounded solution that complements MLLMs’s vision-language reasoning with cognitive spatial intelligence, enabling more capable, adaptable, and cognitively informed AI operating in the real physical world.

### 2 Results

We begin by presenting the BSC-Nav framework, which instantiates braininspired structured spatial memory in embodied agents. We then systematically evaluate its performance across both simulated and real-world scenarios, highlighting: (i) universal navigation capabilities in foundational tasks; (ii)

a Artificial Intelligence

###### b

[Figure 1]

[Figure 2]

Biological Intelligence

The Spatial Cognitive System in Biological Brain

Brain-inspired Spatial Cognition for Navigation (BSC-Nav)

Environment Observation

Updates Surprise-driven Updates

Pose Cam. Intrinsic

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

(",$,%)

###### Neocortex

Surprise Vision Features

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

CortexPrefrontal

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

Motor cortex

RGB Depth

[Figure 17]

[Figure 18]

[Figure 19]

Visual Cortex

Update

[Figure 20]

!

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

… … …

[Figure 28]

Hippocampus

###### Spatial Knowledge

Landmark Contexts Voxelized Feature Space

Voxelized Buffer

Route Landmarks Knowledge

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

Environment Observation

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

Landmark Memory Cognitive Map

###### Projection

###### Incorporate

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

###### Working Memory

[Figure 45]

[Figure 46]

Association Enhancement

[Figure 47]

Text Goal

[Figure 48]

Query

Survey Knowledge

Association Imagination

Multi-modal Retrieval

[Figure 49]

“Move to a Full Closed Barbecue Grill”

[Figure 50]

[Figure 51]

Query

or

A stainless steel barbecue grill fully closed and prominently positioned on a spacious outdoor patio.

Retrieve

[Figure 52]

[Figure 53]

[Figure 54]

Explore Sequence Planning

Candidate Coordinate

The grill’s gleaming surface reflects the warm sunlight, highlighting its sleek lines and sturdy construction.

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

Image Goal

Working Memory

c

Performing Navigation and Higher-level Spatial-aware Skills Based on BSC-Nav

[Figure 60]

[Figure 61]

[Figure 62]

Instruction

[Figure 63]

Manipulation

[Figure 64]

[Figure 65]

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

Embodied QA

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

Explore Sequence Planning

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

MLLM

Low-level Planning Goal Verification Goal Affordance Higher-level Skills

- Fig. 1 The BSC-Nav framework for cognitive spatial intelligence. a, Structured spatial memory in biological brains, comprising landmarks, route knowledge, and survey knowledge. b, The BSC-Nav framework instantiates structured spatial memory in embodied agents. Environment observations (RGB-D images and agent poses) are processed by (i) a landmark memory module encodes and retrieves durable associations of multi-modal environmental cues as the landmarks; and (ii) a cognitive map module accumulates and organizes movement trajectories as the route knowledge into allocentric, map-like representations as the survey knowledge. Upon task invocation, (iii) a working memory module dynamically composes relevant spatial knowledge for adaptive planning and reasoning. c, Structured spatial memory enables not only universal navigation but also higher-level spatial-aware skills.

higher-level spatially-ware skills in instruction-driven visual-language navigation and active embodied question answering; and (iii) real-world efficacy in navigation and mobile manipulation.

#### 2.1 Construction and exploitation of structured spatial memory in embodied agents

Structured spatial memory is essential for guiding goal-directed behavior in complex real-world environments [3–5], yet remains largely absent from current embodied agents. Drawing inspiration from the robust spatial representations in biological systems, namely, landmarks [23, 24], route knowledge [25], and survey knowledge [26], BSC-Nav implements a modular architecture that explicitly constructs and exploits analogous memory structures to achieve spatial cognition capabilities (Fig. 1b, detailed in Methods).

BSC-Nav comprises two branches that are continuously updated to accumulate and organize spatial knowledge during environment exploration. First, a landmark memory module encodes salient environmental cues

as associative triplets comprising spatial coordinates, semantic categories, and contextual descriptions, consolidating landmarks of the surroundings. This design yields abstract and sparse representations that prioritize salient instances, enabling efficient retrieval and forming a flexible, interpretable scaffold of the external space (detailed in Methods). Meanwhile, a cognitive map module transforms egocentric observations along movement trajectories into route knowledge, which are then voxelized into persistent allocentric representations as survey knowledge (detailed in Methods). Inspired by the free-energy principle [32], which posits that biological systems minimize prediction error to refine internal models [32, 33], we implement a surprise-driven update strategy that selectively integrates novel or unexpected spatial observations. Additionally, a voxelized memory buffer maintains diverse spatial representations across viewpoints and timepoints, enhancing robustness and generalization.

To exploit structured spatial memory during task execution, BSC-Nav further incorporates a working memory module that adaptively retrieves landmarks, route knowledge, and survey knowledge for goal localization and trajectory planning (detailed in Methods). Analogous to visual-spatial working memory in biological systems [28, 29], this module dynamically composes spatial representations based on task demands and environmental familiarity.

We propose a hierarchical retrieval strategy guided by the complexity of the goal. For simple targets, the agent leverages MLLMs [22, 34] in a text-only form to reason over semantic associations and contextual cues within the landmark memory (Fig. 1c). For more complex or ambiguous instructions, BSC-Nav activates association-enhanced retrieval over the cognitive map. In this process, MLLMs enrich the initial goal descriptions by inferring object-specific attributes and scene-level priors, transforming vague commands into semantically detailed representations. These enriched descriptions are then rendered into imagined visual prototypes using a text-to-image diffusion model [35]. The generated images are encoded and matched against dense visual features stored in the cognitive map to localize candidate target regions. This hierarchical retrieval strategy allows BSC-Nav to access complementary forms of spatial memory in response to target complexities (category-level and instance-level) and modalities (text and image), supporting precise localization for universal goal-directed navigation (Fig. 2).

Working memory retrieval often produces multiple candidate coordinates, each associated with a confidence score and relative distance. In scenarios with multiple valid targets (e.g., category-level navigation), confidence scores alone may not reliably indicate correctness. Rather than greedily selecting the candidate with the highest confidence, BSC-Nav employs a composite ranking strategy that integrates distance- and confidence-based scores to determine an efficient, adaptive exploration sequence. Low-level movement policies are generated using heuristic planning along this sequence. During execution, MLLMs continuously parse spatial-semantic context from observations, verify landmark proximity, and confirm goal arrival, enabling affordance-aware action generation. Upon reaching the appropriate target locations, agents can invoke

[Figure 96]

[Figure 97]

###### a Working Memory Retrieve Landmark Memory b Working Memory Retrieve Cognitive Map

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

|< Toilet >|
|---|
|Text-Goal|

|< A set of blue classical teapots which placed on a tea tray. >|
|---|
|Text-Goal|

|A set of elegant blue classical teapots, decorated with intricate white floral patterns, resting atop a polished wooden tea tray. The teapots' gentle curves catch the ambient light, creating a soft sheen against their glossy surfaces.|
|---|
|Fine-grained Association|

|[Figure 104]<br><br>[Figure 105]<br><br>[Figure 106]|
|---|
|Visual Imagination|

[Figure 107]

[Figure 108]

Confidence

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

|[Figure 113]|
|---|

[Figure 114]

Area: 64 m!

[Figure 115]

[Figure 116]

[Figure 117]

Confidence

[Figure 118]

[Figure 119]

Area: 288 m!

|[Figure 120]|
|---|

[Figure 121]

|[Figure 122]|
|---|

[Figure 123]

Localizing using Landmark Memory

|[Figure 124]|
|---|

[Figure 125]

Confidence

Localizing using Cognitive Map

|< Bed >|
|---|
|Text-Goal|

|[Figure 126]|
|---|

[Figure 127]

[Figure 128]

|[Figure 129]|
|---|
|Image-Goal|

[Figure 130]

[Figure 131]

[Figure 132]

Area: 217 m!

Area: 64 m!

|[Figure 133]|
|---|

[Figure 134]

[Figure 135]

[Figure 136]

|[Figure 137]|
|---|

|[Figure 138]|
|---|

Localizing using Landmark Memory

Localizing using Cognitive Map

- Fig. 2 Precise localization via hierarchical retrieval in working memory. a, For simple category-level goals, working memory prioritizes retrieval from the landmark memory module for rapid matching and candidate coordinate generation. b, For complex instance-level goals, working memory employs association-enhanced retrieval, converting text instructions into visual features to query the cognitive map module. For image-based goals, the cognitive map is queried directly using extracted visual features.

additional skills, such as grasping objects or answering questions, to accomplish more complex, task-driven objectives (Fig. 1c).

#### 2.2 Universal navigation across modalities and granularities

Through brain-inspired spatial cognition that incorporates the three types of spatial knowledge, BSC-Nav demonstrates outstanding generalization capabilities in navigation tasks, significantly outperforming recent strong baselines in both success rate and efficiency.

We conduct systematic evaluations across 8,195 episodes in 62 indoor scenes from physically reconstructed environments (MP3D [36] and HM3D [37] datasets) in the Habitat simulator [38], covering four representative navigation tasks: Object-Goal Navigation (OGN) [18, 39, 40], OpenVocabulary Object Navigation (OVON) [41], Text-Instance Navigation (TIN) [42], and Image-Instance Navigation (IIN) [43]. In each episode, agents initialized at random positions execute discrete actions (forward 25 cm, turn left 30°, turn right 30° and stop) [44]. Following standard protocols [44], navigation succeeds only when agents execute stop within 1.0 m of goals. Evaluation metrics include the Success Rate (SR) [45] for efficacy and the Success weighted by Path Length (SPL) [45] for efficiency (detailed in Methods). We compare BSC-Nav with end-to-end methods (PixNav [11],

###### c

- a
- b

Category-level Navigation

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

Step=0 Step=162

|[Figure 148]|
|---|

|[Figure 149]|
|---|

|[Figure 150]|
|---|

|[Figure 151]|
|---|

|[Figure 152]|
|---|

Open-Vocabulary Object Navigation

[Figure 153]

[Figure 154]

[Figure 155]

Object-Goal Navigation

Step=8

" Move to a < kitchen lower cabinet >"

[Figure 156]

[Figure 157]

[Figure 158]

" Move to a < sofa >"

[Figure 159]

[Figure 160]

|[Figure 161]|
|---|

|[Figure 162]|
|---|

|[Figure 163]|
|---|

|[Figure 164]|
|---|

|[Figure 165]|
|---|

[Figure 166]

[Figure 167]

[Figure 168]

|Success ! Dist. To Goal: 0.135m<br><br>SPL: 0.457|
|---|

[Figure 169]

###### Candidate Location A Candidate Location B

|Object Goal Navigation Move to a < sofa >|
|---|

MP3D HM3D

HM3D-Uneen HM3D-Seen

100

60

There appears to be a…, which can be identified as a sofa. The sofa appears to be quite close… within 2 meters.

The observation image primarily shows…, There are no visible objects or furniture, such as a sofa, present… we can conclude that the robot has not arrived near the target object.

[Figure 170]

[Figure 171]

78.5

50

80

40.2 38.5

38.9

37.1 35.2

40

35.2

56.5

60

53.1 52.6 54.5

SR(%)

SR(%)

30

41.0

40

34.4

20

25.5

[Figure 172]

[Figure 173]

[Figure 174]

20

10

[Figure 175]

[Figure 176]

0

0

|Success ! Dist. To Goal: 0.106m<br><br>SPL: 0.668|
|---|

DAgRLVLFMBSC-Nav DAgRLVLFMBSC-Nav

PixNavVLFMUniGoalBSC-Nav PixNavVLFMUniGoalBSC-Nav

|[Figure 177]|
|---|

|[Figure 178]|
|---|

|[Figure 179]|
|---|

|[Figure 180]|
|---|

|[Figure 181]|
|---|

[Figure 182]

MP3D HM3D

HM3D-Uneen HM3D-Seen

60

|Text Instance Goal Navigation<br><br>Move to < the rack of chairs that has a black back and white cushion >|
|---|

50

[Figure 183]

47.7

50

40

[Figure 184]

[Figure 185]

[Figure 186]

35.7

32.5

40

|[Figure 187]|
|---|

|[Figure 188]|
|---|

|[Figure 189]|
|---|

|[Figure 190]|
|---|

|[Figure 191]|
|---|

SPL(%)

SPL(%)

30

30.2 28.6 30.4

[Figure 192]

[Figure 193]

30

25.1

21.1

Step=0

19.8 19.6

18.6

20

20

16.7 16.4

[Figure 194]

[Figure 195]

12.5

[Figure 196]

10

10

0

0

[Figure 197]

[Figure 198]

DAgRLVLFMBSC-Nav DAgRLVLFMBSC-Nav

Step=169

PixNavVLFMUniGoalBSC-Nav PixNavVLFMUniGoalBSC-Nav

[Figure 199]

###### Candidate Location A Candidate Location B

[Figure 200]

[Figure 201]

[Figure 202]

The provided observation… appears to be a part of a painting or abstract artwork. Thus, the robot has not arrived at the target location.

[Figure 203]

There are chairs... They have white cushions and black backs, which fit the description..., the target is present in the image with 1 meter.

[Figure 204]

Step=89

Instance-level Navigation

[Figure 205]

[Figure 206]

[Figure 207]

Text Instance Goal Navigation

Image Instance Goal Navigation

[Figure 208]

|[Figure 209]|
|---|

|[Figure 210]|
|---|

|[Figure 211]|
|---|

|[Figure 212]|
|---|

|[Figure 213]|
|---|

" Move to < double bed and its " Move to < > " material is a bamboo frame > "

[Figure 214]

[Figure 215]

[Figure 216]

[Figure 217]

[Figure 218]

Step=52

[Figure 219]

[Figure 220]

[Figure 221]

| |
|---|

[Figure 222]

[Figure 223]

[Figure 224]

[Figure 225]

|[Figure 226]|
|---|

|[Figure 227]|
|---|

|[Figure 228]|
|---|

|[Figure 229]|
|---|

|[Figure 230]|
|---|

[Figure 231]

[Figure 232]

|Success ! Dist. To Goal: 0.514m<br><br>SPL: 0.875|
|---|

[Figure 233]

[Figure 234]

[Figure 235]

HM3D

HM3D

60

100

###### Candidate Location A

Step=0

50

|Open-Vocabulary Object Navigation<br><br>Move to a < Plush Toy >|
|---|

44.9

80

71.4

40

60.2

In the provided image, I can see a plush toy located on the metal basket mounted on the wall. It appears to be… within a distance of approximately 1 to 2 meters from the observation point...

[Figure 236]

56.1

SR(%)

60

SR(%)

30

37.4

20.2

40

17.0 16.5

20

20

[Figure 237]

[Figure 238]

10

[Figure 239]

[Figure 240]

0

0

GOAT PSLUniGoalBSC-Nav

GOATMOD-IINUniGoalBSC-Nav

|Success ! Dist. To Goal: 0.101m<br><br>SPL: 0.830|
|---|

|[Figure 241]|
|---|

|[Figure 242]|
|---|

|[Figure 243]|
|---|

[Figure 244]

[Figure 245]

[Figure 246]

[Figure 247]

[Figure 248]

HM3D

HM3D

Step=105

60

80

Agent

Starting

[Figure 249]

|[Figure 250]|
|---|

50

|[Figure 251]|
|---|

|[Figure 252]|
|---|

|[Figure 253]|
|---|

57.2

[Figure 254]

60

[Figure 255]

40

SPL(%)

SPL(%)

32.5

[Figure 256]

[Figure 257]

30

40

[Figure 258]

[Figure 259]

Success Judgement

[Figure 260]

Candidate

23.3 23.7

20

|[Figure 261]<br><br>[Figure 262]<br><br>Image Instance Goal Navigation|
|---|

16.1

20

[Figure 263]

11.4

[Figure 264]

8.8 7.5

[Figure 265]

10

###### Candidate Location A

Step=0

0

0

GOAT PSLUniGoalBSC-Nav

GOATMOD-IINUniGoalBSC-Nav

The key visual elements, such as the sofa, staircase, and the red wall, align closely in both images, confirming that the robot is at the target location..., and there is no need to move forward to get closer.

[Figure 266]

Goal

Path

- Fig. 3 Goal-directed multi-modal navigation. a, Category-level navigation tasks, including object-goal navigation and open-vocabulary object navigation. b, Instance-level navigation tasks, including text-instance navigation and image-instance navigation. c, Visualization of BSC-Nav navigation trajectories, including the agent’s egocentric observations and target verification via MLLM interaction.

DAgRL [41], PSL [42]) and modular methods that instantiate only analogous landmark memory (VLFM [13], MOD-IIN [46], UniGoal [47], GOAT [48]).

As shown in Fig. 3a, BSC-Nav demonstrates superior spatial generalization in category-level navigation (OGN and OVON). For common targets in OGN tasks, BSC-Nav achieves 78.5% SR on HM3D (6 categories) and 56.5% on MP3D (20 categories with larger areas), surpassing the state-ofthe-art method UniGoal by 24.0% and 15.5%, respectively. Unlike UniGoal, which organizes only landmark memory with abstract goals and scene graphs, BSC-Nav leverages structured spatial memory and thus obtains significant performance gains. In the more challenging OVON tasks involving 79 everyday objects (e.g., “kitchen lower cabinet”), BSC-Nav maintains 40.2% and 38.9% SR on MP3D’s unseen and seen validation sets, outperforming the supervised method DAgRL in zero-shot settings. The results of instance-level navigation (TIN and IIN) further validate BSC-Nav’s robust multi-modal generalization (Fig. 3b). In TIN tasks, it nearly doubles the SR compared to UniGoal and

VLFM. In IIN tasks, direct matching of target images with cognitive map features yields 71.4% SR, outperforming UniGoal by 11.4%.

Notably, BSC-Nav achieves significantly higher navigation efficiency across all tasks compared to baseline methods, as measured by SPL scores. This improvement stems from our distance- and confidence-based scoring strategy within the working memory module, which enables the agent to plan efficient exploration sequences by examining only a small number of candidate locations (Supplementary Fig. 2). Representative visualization examples are presented in Fig. 3c, including top-to-down trajectories, first-person observations, and MLLM-assisted target verification. We provide additional examples in Supplementary Video 1 and benchmark results in Supplementary Fig. 2.

#### 2.3 Higher-level spatially-aware skills

By integrating structured spatial memory with MLLMs’ robust visual grounding and high-level planning capabilities, BSC-Nav demonstrates strong performance on higher-order spatial tasks, such as long-horizon navigation and spatial reasoning based on complex linguistic instructions.

We evaluate BSC-Nav on a representative task termed Long-horizon Instruction-based Navigation (LIN) [49, 50], which requires agents to understand and execute complex instructions containing multiple intermediate goals and spatial constraints. For example, an instruction such as “Go through the glass door, pass between the sofa and the coffee table, walk to the refrigerator, then turn right and stop at the staircase entrance” demands nuanced vision-language reasoning and spatial understanding. To address this, we equip BSC-Nav with GPT-o3 [51], a powerful MLLM with strong reasoning capabilities. GPT-o3 decomposes complex instructions into spatially-grounded subgoals (termed waypoints, such as “glass door”, “refrigerator”, and “staircase entrance”) by language instruction and initial visual observation of the agent. This hierarchical planning strategy transforms instruction-following into sequential goal-directed navigation steps, enabling BSC-Nav to reliably reach each waypoint and ultimately the final destination.

We evaluate BSC-Nav on the VLN-CE Room-to-Room (R2R) benchmark [50], which includes 1,000 human-annotated long-horizon instructions from MP3D. As shown in Fig. 4a, BSC-Nav achieves 38.5% SR in zero-shot settings, only 8.5% below Uni-Navid [12], the state-of-the-art Vision-LanguageAction (VLA) model trained with extensive task-specific supervision. Notably, BSC-Nav attains 53.1% SPL for navigation efficiency, significantly outperforming all baseline methods. These results underscore the strength of combining structured spatial memory with foundation models for generalizing to complex, long-horizon spatial tasks, without requiring instruction-level supervised training. Representative navigation trajectories are visualized in Fig. 4d, including simplified waypoint descriptions and visual prototypes generated by working memory retrieval. Additional examples are provided in Supplementary Video 2.

The value of spatial cognition extends beyond navigation to spatial reasoning and scene understanding. Here we consider a representative task termed

[Figure 267]

- a
- b
- c AR

- d
- e

[Figure 268]

[Figure 269]

VLN-CE R2R

VLN-CE R2R

Human Instruction

Walk into the hallway and at the top of the stairs turn left and walk into the bedroom. Walk past the right side of the bed and turn right into the closet and all the way through to the bathroom. Stop in front of the toilet in the bathroom.

60

60

[Figure 270]

53.1

47.0

50

50

42.7

[Figure 271]

[Figure 272]

Subgoal-1: Move to the {stairs at the end of the hallway}

Subgoal-4: Move to the {toilet in the bathroom}

Subgoal-2: Move to the {bed in the bedroom}

Subgoal-3: Move to the {closet}

38.5

Sub-goal Planning

37.4

40

40

35.9

|[Figure 273]|
|---|

[Figure 274]

SPL(%)

32.0

SR(%)

[Figure 275]

[Figure 276]

[Figure 277]

[Figure 278]

30.0

[Figure 279]

[Figure 280]

30

30

|[Figure 281]|
|---|

|[Figure 282]|
|---|

|[Figure 283]|
|---|

|[Figure 284]|
|---|

[Figure 285]

Sub-goal Imagination

20

20

[Figure 286]

[Figure 287]

[Figure 288]

[Figure 289]

[Figure 290]

[Figure 291]

[Figure 292]

[Figure 293]

10

10

[Figure 294]

[Figure 295]

[Figure 296]

[Figure 297]

[Figure 298]

|[Figure 299]|
|---|

|[Figure 300]|
|---|

|[Figure 301]|
|---|

|[Figure 302]|
|---|

|[Figure 303]|
|---|

|[Figure 304]|
|---|

|[Figure 305]|
|---|

0

0

[Figure 306]

[Figure 307]

CMA NaVidUni-NaVidBSC-Nav

CMA NaVidUni-NaVidBSC-Nav

|Success !<br><br>Dist. To Goal: 0.810m<br><br>SPL: 0.526 Total Step: 315|
|---|

[Figure 308]

[Figure 309]

|[Figure 310]|
|---|

|[Figure 311]|
|---|

|[Figure 312]|
|---|

|[Figure 313]|
|---|

|[Figure 314]|
|---|

|[Figure 315]|
|---|

|[Figure 316]|
|---|

[Figure 317]

[Figure 318]

Human Instruction

Leave the fitness area, and go left. Take the first right at the wine room. Walk straight ahead into the TV room. Stop once you enter the room.

OpenEQA-184subset

[Figure 319]

[Figure 320]

[Figure 321]

Blind LLMs

Question-Agnostic Exploration

Active Exploration

Sub-goal Planning

Subgoal-1: Move to the {exit of the fitness area}

Subgoal-2: Move to the {wine room}

Subgoal-3: Move to the {TV room}

[Figure 322]

|[Figure 323]|
|---|

|[Figure 324]|
|---|

|[Figure 325]|
|---|

Sub-goal Imagination

|[Figure 326]|
|---|

LLM-MatchScore

60

[Figure 327]

[Figure 328]

54.6

46.9

[Figure 329]

[Figure 330]

[Figure 331]

[Figure 332]

[Figure 333]

[Figure 334]

41.8

40

35.5 35.9 34.4 34.2

|[Figure 335]|
|---|

|[Figure 336]|
|---|

|[Figure 337]|
|---|

|[Figure 338]|
|---|

|[Figure 339]|
|---|

|[Figure 340]|
|---|

|[Figure 341]|
|---|

[Figure 342]

[Figure 343]

[Figure 344]

[Figure 345]

[Figure 346]

[Figure 347]

20

|Success !<br><br>Dist. To Goal: 1.272m SPL: 0.769 Total Step: 165|
|---|

[Figure 348]

|[Figure 349]|
|---|

|[Figure 350]|
|---|

|[Figure 351]|
|---|

|[Figure 352]|
|---|

|[Figure 353]|
|---|

|[Figure 354]|
|---|

|[Figure 355]|
|---|

[Figure 356]

[Figure 357]

[Figure 358]

0

GPT-4vGPT-4o CGCap.SVMCap.Multi-FrameExplore-EQABSC-Nav

[Figure 359]

[Figure 360]

[Figure 361]

[Figure 362]

[Figure 363]

[Figure 364]

[Figure 365]

Now, we need to go to {A bed in the bedroom.}

[Figure 366]

[Figure 367]

BSC-Nav Yes, the bedroom door is open.

Is the bedroom door open?

|[Figure 368]|
|---|

|[Figure 369]|
|---|

[Figure 370]

[Figure 371]

[Figure 372]

……

……

OL

SU

[Figure 373]

[Figure 374]

[Figure 375]

[Figure 376]

[Figure 377]

[Figure 378]

Now, we need to go to {An upstairs wall featuring four paintings.}

[Figure 379]

What is underneath the four paintings upstairs?

[Figure 380]

A radiator is underneath the four paintings upstairs.

OR

[Figure 381]

BSC-Nav

[Figure 382]

[Figure 383]

| |
|---|

| |
|---|

[Figure 384]

[Figure 385]

[Figure 386]

OSR

……

……

WK

FR

- Fig. 4 Higher-level spatially-aware skills enabled by BSC-Nav. a, Comparison of BSC-Nav and baseline methods on long-horizon instruction-following tasks using the VLNCE R2R benchmark. b, Comparison of BSC-Nav and baseline methods on active embodied question answering tasks using the A-EQA benchmark. c, Category-wise performance breakdown across seven question types for BSC-Nav, baselines, and humans in embodied question answering tasks. d, e, Representative examples of BSC-Nav’s navigation trajectories in human instruction navigation (d) and embodied question answering (e) tasks.

Active Embodied Question Answering (A-EQA) [52–54], which requires agents to answer spatially grounded questions through active exploration of the environment. Unlike static visual question answering, A-EQA demands spatial understanding, exploratory planning, and dynamic observation synthesis, all well aligned with BSC-Nav’s strengths. To address each question, BSC-Nav first parses target waypoints (instances or regions relevant to the query) and then conducts goal-directed exploration. Upon reaching the appropriate locations, GPT-4o [34] integrates local observations with the original question to generate spatially grounded answers.

We evaluate BSC-Nav on the A-EQA subset of the OpenEQA benchmark [54], which includes 184 questions that span seven categories: Object Recognition (OR), Object Localization (OL), Attribute Recognition (AR), Spatial Understanding (SU), Object State Recognition (OSR), Functional Reasoning (FR), and World Knowledge (WK). We compare BSC-Nav with

three representative baselines: a blind LLM that answers without visual observations; a question-agnostic frontier exploration strategy [54]; and ExploreEQA [55], an active exploration method without structured spatial memory. The overall performance is measured by LLM-Match [54], which computes semantic similarity between the generated and reference answers using LLMs (detailed in Methods). As shown in Fig. 4b, BSC-Nav achieves the highest LLM-Match score of 54.6, significantly outperforming all baselines. The category-wise breakdown in Fig. 4c highlights especially strong improvements in tasks that require fine-grained spatial localization (OL and OSR) for rapid and accurate exploration. Despite these advances, a performance gap remains compared to that in humans, underscoring the ongoing challenges of embodied spatial cognition. Representative examples in Fig. 4e illustrate how BSC-Nav combines goal-directed exploration with structured spatial memory to address diverse spatial reasoning problems.

#### 2.4 Real-world navigation and mobile manipulation

To demonstrate BSC-Nav’s generalization beyond simulation and its broad applicability in downstream tasks, we deploy it on a custom-built mobile robotic platform within physical indoor environments. This evaluation illustrates how structured spatial memory enables reliable long-range navigation and integrated manipulation in the real physical world. The mobile platform (Fig. 5a) features a compact mechanical design equipped with high-precision motion control, as well as modular sensory and actuation capabilities. We conduct a total of 75 navigation episodes in a two-story indoor space (around 200 m2, Fig. 5b), covering three types of goal-directed tasks: Object-Goal Navigation (OGN), Text-Instance Navigation (TIN), and Image-Instance Navigation (IIN), with 5 distinct targets per task (Fig. 5c). Each target is tested over 5 trials from randomly sampled starting positions, with an average path length of 23.4 m.

BSC-Nav demonstrates robust spatial generalization in all real-world tasks. As shown in Fig. 5d, it achieves at least 3 out of 5 successful trials per target (SR is defined as reaching within 1.0 m), with IIN performing best, reaching 100% SR on 4 out of 5 targets. Both OGN and TIN reach 100% SR on 2 targets and exceeds 66.7% SR across all cases. Even in failure cases, BSC-Nav reliably localizes to semantically plausible regions, demonstrating strong spatial awareness. The Distance-to-Goal (DtG) distribution in Fig. 5e further confirms the precise stop location, with the final DtG below 2.5 m in all cases and tight clustering between targets, particularly for IIN. Navigation efficiency is reflected in a mean velocity of 0.76 m/s with low variance (Fig. 5f), indicating stable and efficient movement. Representative navigation trajectories are visualized in Fig. 5g–h, with complete demonstrations available in Supplementary Videos 3–7.

Beyond navigation, BSC-Nav’s structured spatial memory supports robust mobile manipulation guided by natural language. Although prior embodied manipulation systems [56–58] have shown promise in static, small-scale

[Figure 387]

a b

[Figure 388]

[Figure 389]

[Figure 390]

[Figure 391]

[Figure 392]

[Figure 393]

[Figure 394]

[Figure 395]

[Figure 396]

Robot Overview

RGBD Camera

[Figure 397]

87°

Intel RealSense d435i

[Figure 398]

[Figure 399]

[Figure 400]

[Figure 401]

[Figure 402]

###### Floor-2Floor-1

Industrial Battery Computer

SLAM Kit

LiDAR, IMU

Nvidia RTX4090

7-DoF Robotic Arm

[Figure 403]

[Figure 404]

Franka Research 3

1528mm

[Figure 405]

[Figure 406]

[Figure 407]

[Figure 408]

[Figure 409]

[Figure 410]

Omnidirectional Locomotion Chassis

AGILEX Ranger-mini-3.0

Ackerman Steering

[Figure 411]

[Figure 412]

[Figure 413]

[Figure 414]

[Figure 415]

[Figure 416]

[Figure 417]

[Figure 418]

[Figure 419]

[Figure 420]

[Figure 421]

[Figure 422]

[Figure 423]

[Figure 424]

[Figure 425]

494mm 773mm

[Figure 426]

[Figure 427]

- e
- f

c

1 OGN

2 3 4 5

2.5

###### DistancetoGoal(m)

Kitchen island

RealworldGoalsSuccessRate

2.0

Vending Sofa Plant

Table

- Goal-1

- Goal-2

- Goal-3

- Goal-4

- Goal-5

1.5

A seating area with a soft orange cushion

A black water dispenser

A model of the robot WALL-E

A Green lazy sofa

TIN

A yellow armchair

1.0

0.5

[Figure 428]

[Figure 429]

[Figure 430]

[Figure 431]

[Figure 432]

0.0

IIN

OGN

ON TIN IIN

1.5

d

1

2 3 4 5

###### AverageSpeed(m/s)

- Goal-1

- Goal-2

- Goal-3

- Goal-4

- Goal-5

OGN TIN IIN

5/5 3/5 3/5 4/5 5/5

1.0

.76

3/5 3/5 5/5 3/5 5/5

0.5

3/5 5/5 5/5 5/5 5/5

0.0

ON TIN IIN

OGN

- g
- h

[Figure 433]

[Figure 434]

[Figure 435]

Imaginations

Navigate to { A yellow armchair }

[Figure 436]

[Figure 437]

[Figure 438]

[Figure 439]

[Figure 440]

[Figure 441]

[Figure 442]

[Figure 443]

Top-down Trajectory

[Figure 444]

[Figure 445]

observation Egocentric

Exocentric

[Figure 446]

[Figure 447]

[Figure 448]

[Figure 449]

[Figure 450]

[Figure 451]

Steps

… … … …

[Figure 452]

[Figure 453]

[Figure 454]

[Figure 455]

[Figure 456]

[Figure 457]

[Figure 458]

[Figure 459]

Observation

[Figure 460]

[Figure 461]

[Figure 462]

[Figure 463]

[Figure 464]

[Figure 465]

[Figure 466]

[Figure 467]

|[Figure 468]<br><br>[Figure 469]|
|---|

Navigate to { A toy model of the little robot WALL-E }

Imaginations

[Figure 470]

[Figure 471]

[Figure 472]

[Figure 473]

[Figure 474]

[Figure 475]

[Figure 476]

[Figure 477]

[Figure 478]

Top-down Trajectory

[Figure 479]

[Figure 480]

[Figure 481]

observation Egocentric

Exocentric

[Figure 482]

[Figure 483]

[Figure 484]

[Figure 485]

Steps

… … … …

[Figure 486]

[Figure 487]

[Figure 488]

[Figure 489]

[Figure 490]

[Figure 491]

[Figure 492]

[Figure 493]

[Figure 494]

Observation

[Figure 495]

[Figure 496]

[Figure 497]

- Fig. 5 Universal navigation capabilities in real-world scenarios. a, The custombuilt robotic platform integrates perception, navigation, and manipulation capabilities. b, The indoor experimental environment spans 200 m2 and includes diverse functional zones (e.g., office, lounge, reception, and kitchen). c-f, Real-world navigation performance across object-goal, text-instance, and image-instance tasks with 15 distinct targets (c). For each target, we report SR from 5 trials with randomized starting positions (d), final distance to goal (e), and mean navigation velocity (f). g, h, Representative examples of real-world navigation. Each shows a top-down trajectory with timestamps (left) and corresponding egocentric/allocentric views (right).

environments, they often lack the spatial generalization necessary for largescale deployment. BSC-Nav overcomes this limitation by seamlessly integrating long-horizon navigation with goal-conditioned manipulation. In our demonstrations, natural language instructions are interpreted by GPT-4 [59] to generate waypoint-action sequences. Upon reaching each waypoint, the agent executes a predefined manipulation primitive (such as grasp, place, pour, and

[Figure 498]

###### a

[Figure 499]

###### Go to the marble table next to the shredder and clean the stains with the rag.

[Figure 500]

[Figure 501]

Navigation: < Marble table next to the shredder > Manipulation: < Clean the stains with the rag >

[Figure 502]

[Figure 503]

[Figure 504]

[Figure 505]

[Figure 506]

1 2 3 4 5

[Figure 507]

[Figure 508]

[Figure 509]

[Figure 510]

[Figure 511]

[Figure 512]

[Figure 513]

[Figure 514]

[Figure 515]

###### b

[Figure 516]

###### Transfer the cookie box from the white table to the kitchen island.

[Figure 517]

[Figure 518]

Navigation: < Cookie box on the white table >

Manipulation: < Grasp cookie box >

[Figure 519]

[Figure 520]

[Figure 521]

[Figure 522]

[Figure 523]

1 2 3 4 5

[Figure 524]

[Figure 525]

[Figure 526]

[Figure 527]

[Figure 528]

[Figure 529]

[Figure 530]

[Figure 531]

[Figure 532]

Manipulation: < Drop cookie box > Navigation: < Cookie box on the white table >

[Figure 533]

[Figure 534]

[Figure 535]

[Figure 536]

[Figure 537]

[Figure 538]

10 9 8 7 6

[Figure 539]

[Figure 540]

[Figure 541]

[Figure 542]

[Figure 543]

[Figure 544]

[Figure 545]

[Figure 546]

c

[Figure 547]

###### Please make breakfast on the plate under the white table. I need some oats, coco balls, and some milk.

[Figure 548]

[Figure 549]

Navigation: < Oatmeal jar > Manipulation: < Pour oats onto the plate >

[Figure 550]

[Figure 551]

[Figure 552]

[Figure 553]

[Figure 554]

[Figure 555]

1 2 3 4 5

[Figure 556]

[Figure 557]

[Figure 558]

[Figure 559]

[Figure 560]

[Figure 561]

[Figure 562]

[Figure 563]

Navigation: < Plate on the white table > Manipulation: < Grasp coco balls > Navigation: < Coco balls >

[Figure 564]

[Figure 565]

[Figure 566]

[Figure 567]

[Figure 568]

[Figure 569]

- 10 9 8 7 6

- 11 12 13 14 15

[Figure 570]

[Figure 571]

[Figure 572]

[Figure 573]

[Figure 574]

[Figure 575]

[Figure 576]

[Figure 577]

[Figure 578]

[Figure 579]

Manipulation: < Pour coco balls onto the plate > Navigation: < Milk bottle > Manipulation: < Grasp milk bottle >

[Figure 580]

[Figure 581]

[Figure 582]

[Figure 583]

[Figure 584]

[Figure 585]

[Figure 586]

[Figure 587]

[Figure 588]

[Figure 589]

[Figure 590]

[Figure 591]

[Figure 592]

Manipulation: < Pour milk onto the plate > Navigation: < Plate on the white table >

[Figure 593]

[Figure 594]

[Figure 595]

[Figure 596]

[Figure 597]

[Figure 598]

[Figure 599]

20 19 18 17 16

[Figure 600]

[Figure 601]

- Fig. 6 Real-world mobile manipulation. The coordination between spatial memorydriven navigation and manipulation primitives enables the execution of long-horizon tasks specified through human instructions. a, Single-waypoint task, cleaning stains on a marble table next to a shredder. b, Object transfer task, relocating a cookie box from table to the kitchen island, requiring navigation between manipulation actions. c, Complex multistep task, preparing breakfast by sequentially navigating to and manipulating three openvocabulary targets (oatmeal jar, coco balls, and milk bottle) to assemble ingredients on a plate. Full demonstration are provided in Supplementary video 8-10.

so on). Representative examples in Fig.6 show both single-step and multistep tasks. Notably, the “make breakfast” task (Fig. 6c) involves identifying and interacting with three spatially distributed, open-vocabulary objects, emphasizing the critical role of structured spatial memory in long-horizon reasoning and reliable object grounding. Full demonstrations are provided in Supplementary Videos 8–10.

### 3 Discussion

This work demonstrates that biological principles of spatial cognition can be effectively instantiated in embodied agents. By constructing a structured spatial memory comprising landmarks, route knowledge, and survey knowledge, BSC-Nav advances not only navigation performance but also the emergence of cognitive spatial intelligence, which complements the powerful perceptual and reasoning capabilities of modern MLLMs in realizing general-purpose embodied AI systems.

###### From reactive behavior to cognitive spatial intelligence

Conventional embodied agents often rely on reinforcement learning or imitation learning, acquiring task-specific policies through extensive trial-and-error or teleoperated demonstrations. While effective in controlled settings, these methods remain fundamentally reactive, responding to immediate stimuli without persistent or reusable spatial knowledge. BSC-Nav addresses this limitation by integrating structured spatial memory with foundation models, enabling a transition from observation-driven pattern matching to multi-level spatial reasoning. Empirically, BSC-Nav exhibits strong performance across navigation tasks of varying modalities and granularities, especially in open-vocabulary and instance-level settings. For example, it achieves 38.5% SR on the challenging VLN-CE benchmark in a zero-shot setting, coming within 8.5% of the leading supervised method, while attaining superior efficiency. These results demonstrate how memory-centric spatial representations enable embodied agents to decouple planning from perception, reuse prior experience across tasks, and translate high-level goals into concrete actions, exemplifying the hallmarks of cognitive spatial intelligence.

###### Computational instantiation of biological spatial cognition

BSC-Nav provides a computational framework that operationalizes longstanding theories of spatial knowledge in biological systems [16, 17]. While neuroscience has proposed that spatial cognition builds upon interconnected representations of landmarks, route knowledge, and survey knowledge, most experimental evidence has been behavioral or correlational. BSC-Nav demonstrates how these elements can be synergistically implemented to support long-horizon spatial understanding, planning, and reasoning. In particular, the cognitive map module employs a surprise-driven update strategy inspired by the free-energy principle [32], aligning with the hypothesis that biological brains refine internal models by minimizing prediction errors [32, 33]. Furthermore, recent findings suggest that sequential route learning drives the formation of allocentric cognitive maps in the hippocampus-entorhinal circuits [60], a process mirrored by BSC-Nav’s trajectory voxelization. These parallels reveal a convergent computational basis shared between cognitive science and AI.

###### Toward an embodied Turing test [61] for spatial cognition

As embodied AI systems advance in perceptual and motor capabilities, it becomes increasingly important to evaluate their spatial understanding in realworld contexts. Analogous to the classic Turing test in language, one may envision a benchmark for spatial cognition that evaluates whether embodied agents exhibit human-like competence for goal-directed abstraction, planning, and reasoning grounded in physical space. Our study highlights some foundational dimensions: (i) real-time construction of reusable spatial representations; (ii) abstraction and reasoning from sparse, partial observations; and (iii) translation of high-level goals into actionable spatial plans. While BSC-Nav exhibits remarkable progress across these dimensions, performance gaps remain. For instance, BSC-Nav achieves a 54.6 LLM-Match score on A-EQA, significantly outperforming prior methods but still trailing human performance by 27.5%. This underscores the complexity of human spatial cognition, which integrates commonsense knowledge, causal inference, and abstraction from minimal environmental cues. Future benchmarks may formalize a comprehensive embodied Turing test [61] for spatial intelligence, incorporating challenges such as adaptation to environmental changes, multi-step route narration, and collaborative problem-solving.

###### Outlook and future directions

BSC-Nav demonstrates that bio-inspired structured spatial memory can substantially enhance generalization and adaptability in embodied AI systems. Future work may focus on scaling the framework to dynamic and unstructured settings, enhancing memory efficiency for real-time deployment, supporting collaborative multi-agent interactions, and incorporating more available sensory modalities. Beyond navigation, this architecture lays the groundwork for broader cognitive functions through hierarchical memory organization, resembling how biological systems coordinate perception, cognition, and decision-making. As current MLLMs-based embodied agents exhibit limited spatial competence, BSC-Nav underscores the promise of memory-centric design in closing this gap. While the pursuit of AGI remains a long-term objective, such advances offer tangible progress toward more capable, adaptable, and cognitively informed AI in the real physical world.

### 4 Methods

#### 4.1 The BSC-Nav framework

This section describes the implementation details of BSC-Nav, including the three synergistic modules (Fig. 1b) and the pipeline of constructing and leveraging structured spatial memory (Supplementary Fig. 1).

###### Observation space

The observation space of BSC-Nav is defined as Ot = {It,Dt,Pt}, comprising the RGB image It ∈ RH×W×3, the depth image Dt ∈ Rw×h and the pose of the agent Pt = (Xt,Yt,ϕat ) ∈ R3 at time t, where Xt and Yt denote the agent’s 2D position in the world coordinate system (defaulting to the Cartesian coordinate centered at the SLAM [62] initialization point), ϕt is the angle of yaw, and H and W are image width and height, respectively. For simplicity, we omit the time subscript t in subsequent descriptions. RGB images serve as input for both salient object detection and visual feature extraction, while depth images and pose information enable projection from pixel coordinates to the world coordinate system.

###### Landmark memory

During active exploration, an observation O is processed by two branches in parallel to construct structured spatial memory (Fig. 2a). For landmark memory, we instantiate it as a list of 4-tuples as follows:

Mlandmark = {Lk}Nk=1, Lk = {ck,θk,ρk,Tk}, (1)

where θk = (Xkw,Ykw,Zkw) ∈ R3 represents the 3D coordinates of the center of the k-th instance in the world coordinate system, ck ∈ C denotes the openvocabulary category of the predefined category set C, ρk ∈ [0,1] indicates detection confidence, and Tk is the description of the instance generated by GPT-4o [34] based on observations, which encompasses texture, shape, and spatial contextual semantics.

To obtain the world coordinates θk, we perform a series of coordinate transformations. Given a detected object with the bounding box center at pixel coordinates (uk,vk) and the corresponding depth value dk, we first compute the 3D point in the camera coordinate system, which is defined with its origin at the camera’s optical center, the xy-plane parallel to the image plane, and the z-axis aligned with the camera’s optical axis. The point in the camera coordinate system can be calculated through inverse perspective projection:

 

  = dk

 

 , (2)

(uk − cx)/fx (vk − cy)/fy 1

uk vk

pcamk = dk · K−1

1

where K ∈ R3×3 is the intrinsic matrix of the camera with focal lengths (fx,fy) and principal point (cx,cy). Subsequently, we transform this point into the world coordinate system. Given the robot pose Pt = (Xt,Yt,ϕt) from the observation space at time t, we construct the base-to-world transformation

matrix:

  

   ∈ SE(3), (3)

cosϕt −sinϕt 0 Xt sinϕt cosϕt 0 Yt

Tbaseworld =

0 0 1 Zbase 0 0 0 1

where Zbase represents the robot base height in the world frame. The point is then transformed through cascaded homogeneous transformation:

pworldk 1

= TbaseworldTcambase

pcamk 1

, (4)

where Tcambase ∈ SE(3) represents the fixed rigid transformation from camera to robot base frame. The final world coordinates are extracted as θk = pworldk = (Xkw,Ykw,Zkw)⊤. In practice, we employ the open-vocabulary object detector YOLO-World [63] for the perception of salient instances. We predefine several common object categories as the landmark category set C = {“sofa”,“sink”,“bed”,...}, with a detection confidence threshold to exclude semantically ambiguous and non-salient instances.

To prevent duplicate memorization of the same instance, we perform overlap detection for each newly detected landmark LN+1. Define the spatial overlap set as: U = {Lj ∈ Mlandmark : ∥θN+1 − θj∥2 < δoverlap ∧ cN+1 = cj}. If U ̸= ∅, we perform the memory fusion of existing landmarks:

 

cfused = cN+1 θfused = ρN+1ρ·θN+1+ j∈U ρj·θj

N+1+ j∈U ρj ρfused = ρN+1|U|+ +1 j∈U ρj Tfused = Tk, k = arg maxj∈{N+1}∪U ρj.

Lfused =

(5)



After memory fusion, all elements in U are removed from Mlandmark and Lfused is added. This update ensures that each landmark 4-tuple represents a unique spatial instance, avoiding information redundancy.

###### Cognitive map

In parallel with the landmark memory module, the cognitive map module leverages RGB observations from exploration trajectories to continuously project visual cues into voxelized visual-spatial representations, progressively integrating and updating route knowledge beyond previous static and simplified representations (Fig. 2a). We define the cognitive map as a discrete voxelized representation:

Mcog = {Fv}v∈V, where Fv = {fb}Bb=1, fb ∈ Rℏ, (6)

where V ⊆ Z3 denotes the discrete voxel index space, each voxel v = (vx,vy,vz) maintains a feature buffer Fv containing up to B feature vectors and ℏ is the

dimension of the visual feature. To achieve fine-grained visual-spatial encoding, we employ DINO-v2 [30] (a powerful self-supervised visual encoder) to extract patch-level features from continuous 2D RGB observations. Given an RGB image I ∈ RH×W×3, DINO-v2 produces patch tokens organized as a spatial grid Fpatch ∈ RH

′×W′×D, where H′ = H/s and W′ = W/s with patch stride s. These patch-level features are projected from 2D image coordinates to corresponding voxel coordinates through a multi-stage transformation process. Let (i,j) denote the indices in the patch grid where i ∈ {0,1,...,H′ −1} and j ∈ {0,1,...,W′ − 1}. For each patch (i,j), we first determine its center pixel coordinates in the original image:

(uij,vij) = (j · s + s/2,i · s + s/2), (7)

where s is the patch stride. Using the depth value dij sampled at this location, we compute the corresponding 3D point in the camera coordinate system:

 

 . (8)

uij vij 1

pcamij = dij · K−1

This point is then transformed into world coordinates using the robot pose P = (X,Y,ϕ) from the observation space:

pworldij 1

= TbaseworldTcambase

pcamij 1

, (9)

where Tbaseworld is constructed from the robot pose as defined in Eq. (7). With pworldij = (Xijw,Yijw,Zijw)⊤, we discretize the continuous world coordinates into voxel indices:

vij =

Xijw ∆

G 2

+

,

Yijw ∆

G 2

+

,

Zijw ∆

, (10)

where ∆ is the voxel size (spatial resolution) and G is the grid dimension. The offset G/2 centers the voxel grid at the world origin. The visual feature Fpatch[i,j] ∈ RD from patch (i,j) is then associated with voxel vij.

In Mcog, we avoid redundantly storing all visual features from new observations, which would lead to information overload and inefficient retrieval. We also forgo conventional fusion methods, such as grid averaging or distancebased weighting [64, 65], which often introduce bias in visual feature representation. Instead, inspired by biological learning and memory, we maintain dynamic buffers for each voxel and introduce a surprise-based update strategy. Neuroscience research suggests that biological brains update their internal models by minimizing the difference between predicted and observed sensory inputs, known as the free-energy principle [32, 33]. Similarly, BSC-Nav’s cognitive map is updated according to the degree of deviation between new

observations and existing memory in a given spatial region, that is, the level of “surprise”. For each new visual feature fnew projected to voxel v ∈ Z3, we compute its surprise score as

1 FNn(v) fb∈FNn(v)

S(fnew,v) =

D(fnew,fb), (11)

where Nn(v) = {v′ ∈ Z3 : ∥v − v′∥∞ ≤ n} denotes the n-hop cubic neighborhood around voxel v, FNn(v) = v′∈Nn(v) Fv′ represents the union of all feature buffers in the neighborhood, and D(·,·) is the distance metric (e.g., cosine distance). We set a predefined threshold τ that defaults to 0.5. When S(fnew,v) > τ, fnew is added to Fv. If |Fv| = B, the feature with the lowest surprise score within the buffer is replaced to maintain memory efficiency. This surprise-based update strategy achieves two key advantages: (i) it enhances the robustness of spatial knowledge by selectively caching diverse features across viewpoints and timepoints in adapting to dynamic environments; and (ii) it maintains memory storage and retrieval efficiency by avoiding redundant encoding of stable environmental elements.

###### Working memory

The working memory module is responsible for the hierarchical retrieval and strategic reorganization of spatial knowledge from both the landmark memory Mlandmark and the cognitive map Mcog, enabling BSC-Nav to address navigation tasks of varying modalities and granularities (Fig. 2b). Unlike the parallel construction and passive updating of the two memory branches, the working memory is activated only upon receiving navigation instructions. It employs a hierarchical retrieval strategy guided by instruction complexity. For simple and concrete goals, it prioritizes the fast retrieval of landmark memory. For fine-grained or image-based instructions, it further engages the cognitive map for precise visual-spatial localization (Fig. 2c).

MLLM-reasoning retrieval for landmark memory. The structured knowledge base of landmark memory, including landmark categories, confidence scores, and contextual descriptions, provides a strong foundation for MLLMs to reason about target locations. Unlike direct rule matching, this approach enables context-aware reasoning and can infer the location of unrecorded targets based on spatial associations. For example, even if a “toaster” is not explicitly recorded, the system can infer its location by leveraging co-located landmarks such as “stove” and “kitchen island”. Specifically, we design retrieval prompts (see prompt templates in Supplementary Sec. A) that guide a text-only GPT-4 to integrate confidence scores and descriptive semantics to generate a set of candidate coordinates {θcandi }Ki=1 from Mlandmark.

Association-enhanced retrieval for cognitive map. To bridge the modality gap between textual instructions and visual representations, we perform

association-enhanced retrieval over the cognitive map. It first employs textonly GPT-4o to refine goal descriptions, enriching them with detailed texture and spatial context information. Specifically, in LIN tasks, initial visual observations are additionally provided as environmental priors to enhance the accuracy and specificity of refined descriptions. These enriched descriptions are then used to “imagine” potential visual appearances of targets through a text-to-image generation model, defaulting to Stable Diffusion 3.5 [66]. This “imagine-then-localize” process resembles human pre-navigation thinking. The generated imagined image is encoded by DINO-v2 to extract patch-level features {fi}Qi=1, followed by center-distance weighted pooling to obtain the instance visual representation:

N i=1 wi · pi

, where wi = exp(−α · (xi,yi) − (xc,yc)2), (12)

ftarget =

N i=1 wi

where (xi,yi) denotes the spatial coordinates of the i-th patch, (xc,yc) is the image center, and α is the temperature parameter. This weighting strategy suppresses background interference and enhances central features of the target instance. As an intermediate query, the pooled visual features are further matched against features stored in the cognitive map, returning the top-K voxel coordinate set with maximum cosine similarity. We perform similarityweighted DBSCAN clustering [67] on the voxel coordinate set to obtain the grid coordinates of cluster centers as the final spatial coordinate candidates. These grid coordinates need to be further back-projected to the world coordinate system, yielding {θcandi }Qi=1 for subsequent low-level planning.

Exploration sequence planning. We design a composite scoring function that integrates target existence probability and spatial distance to prioritize the set of candidate spatial coordinates. Specifically, for candidates retrieved from the landmark memory, we use detection confidence as the existence probability. For candidates retrieved from the cognitive map, we employ the cosine similarity of visual features. The priority scoring function is defined as:

Hi = λ · pi + (1 − λ) · (1 −

di dmax

), (13)

where pi represents the existence probability (confidence or similarity) of the i-th candidate, di is the Euclidean distance between the candidate and the starting point, dmax = maxj dj normalizes distance across candidates. The weighting hyperparameter λ balances existence probability and exploration efficiency, defaulting to λ = 0.5 as equal importance to both components.

###### Low-level navigation policy generation

Low-level navigation policies are typically derived from deterministic path planning conditioned on environmental constraints and high-level spatial goals [44]. BSC-Nav employs a hierarchical navigation strategy: the exploration sequence of each candidate coordinate serves as a high-level policy, while

heuristic algorithms are used to generate actionable low-level policies toward each candidate coordinate. In simulation environments, we employ the greedy shortest path algorithm provided by the Habitat simulator [38], which operates directly on 3D mesh representations of the scene to compute optimal action sequences within a discrete action space. For real-world deployment, we implement a two-tier planning architecture. Global planning is carried out using the A* algorithm [68] on occupancy grid maps constructed via LiDAR-based SLAM, yielding globally optimal paths. Local planning leverages the Timed Elastic Band (TEB) algorithm [69, 70], which dynamically adjusts trajectories to avoid obstacles while maintaining efficiency. The TEB planner outputs continuous velocity commands to control the robot chassis, ensuring smooth and precise motion execution.

###### Goal verification and affordance

Upon arriving at a candidate coordinate, BSC-Nav verifies whether the navigation target is present at the location. The robot first performs a 360° rotational scan to capture a sequence of RGB images. Cosine similarities are then computed between the CLIP visual embeddings of these images and the goal’s text or visual embeddings to identify the viewing angle with the highest semantic alignment. The selected image is subsequently fed into GPT-4o for precise target verification. Beyond confirming target presence, GPT-4o is also prompted to generate a sequence of affordance-based actions to guide the robot in fine-tuning its pose, adjusting its relative position, and orientation for optimal proximity and visibility. This ensures favorable initial conditions for downstream manipulation tasks.

#### 4.2 Experimental details Simulator and datasets

The simulation experiments are conducted under the Habitat 3.0 platform [71], a commonly used simulation framework for embodied AI and human–robot interaction in domestic environments. We select this platform for its strong support for large-scale embodied navigation, including (i) habitat-sim, a highperformance simulator offering photorealistic rendering and physics-based interactions for mainstream indoor datasets, and (ii) habitat-lab, a modular benchmarking suite supporting a variety of navigation tasks with standardized pipelines and metrics. Here we evaluate BSC-Nav and baseline methods on four foundational navigation tasks:

- 1. Object-Goal Navigation (OGN). This task [72] comprises two benchmarks, including (i) 2,195 episodes across 34 HM3D scenes involving 6 object categories, and (ii) 2,000 episodes across 10 MP3D scenes with 20 object categories.
- 2. Open-Vocabulary Object Navigation (OVON). This task [41] extends to 79 open-vocabulary categories across 10 MP3D scenes to address the limited object categories in object-goal navigation. We sampled 1,000 episodes

- each from the validation-seen and validation-unseen splits. The seen split includes categories present in training (though not exact instances), while the unseen split comprises entirely novel and semantically dissimilar categories.
- 3. Text-Instance Navigation (TIN). This task [42] provides natural language descriptions for 795 instances across 36 HM3D scenes. Descriptions encompass both intrinsic attributes (inherent object properties including shape, color, and material) and extrinsic attributes (surrounding environmental contexts), annotated with a strong MLLM termed CogVLM [73]. We evaluate all 1,000 test episodes.
- 4. Image-Instance Navigation (IIN). This task [43] employs single-view rendered images as navigation targets for instances across 34 HM3D scenes. We sampled 1,000 episodes from the validation split.

Beyond these foundational navigation tasks, we further evaluate BSC-Nav and baseline methods on higher-level spatially-aware skills:

- 1. Long-horizon Instruction-based Navigation (LIN). We employ the VLN-CE Room-to-Room (R2R) benchmark [50], which provides 1,000 navigation episodes in Habitat-lab based on human-annotated long-horizon instructions across 11 MP3D scenes.
- 2. Active Embodied Question Answering (A-EQA). We develop a custom evaluation pipeline within Habitat-lab using OpenEQA [54]. Agents are initialized at the first frame of each recorded exploration trajectory and are allowed to actively explore the environment to answer a given spatially grounded question. We evaluate the 184 test queries that span seven task categories.

###### Evaluation metrics

We adopt two-dimensional metrics [45] to quantify navigation performance. Success Rate (SR) measures the proportion of successful navigation episodes relative to the total test episodes, assessing the agent’s capability to accurately navigate target object instances. Notably, in OGN, any instance of the target category constitutes a valid goal regardless of distance from the starting position, whereas instance-level navigation requires reaching a unique specified instance in the environment. Specifically, an episode is deemed successful if the Euclidean distance between the agent and the target object is within 1.0 m upon executing the “stop” action. While Success weighted by Path Length (SPL) jointly considers navigation success and path efficiency:

N

1 N

SPL =

i=1

L∗i max(Li,L∗

Si ·

, (14)

i )

where N denotes the total number of episodes, Si is the binary success indicator for episode i (1 for success, 0 for failure), L∗i represents the geodesic shortest path length, and Li is the actual trajectory length executed by the

agent. SPL values range from 0 to 1, with higher values indicating more efficient navigation. While SR quantifies task completion capability, SPL further evaluates the optimality of path planning, collectively providing a comprehensive assessment of navigation performance.

For A-EQA tasks, we adopt the LLM-Match scoring metric proposed in Open-EQA [54] to assess the correctness of agent responses. This metric accounts for the open-vocabulary nature of the agent responses, serving as a substitute for manual evaluation. Specifically, given a question Qi, a human annotated response A∗i , and an agent response Ai, the LLM is prompted to assign a score σi ∈ {1,...,5}, where σi = 1 denotes an incorrect answer, σi = 5 denotes a correct answer, and intermediate values reflect varying degrees of partial correctness. The overall LLM-based correctness is then computed as follows:

NQ

σi − 1 4 × 100%, (15)

1 NQ

LLM-Match =

i

where NQ denotes the total number of questions. Following the OpenEQA protocol, we employ text-only GPT-4 under official prompts to ensure evaluation fairness.

###### Hardware stack

We develop an embodied platform for BSC-Nav deployment, comprising five core components: locomotion chassis, industrial computer, SLAM suite, robotic arm, and vision sensors.

The locomotion system employs the Agilex Ranger-mini 3.0 platform, chosen for its balance between cost and payload capacity. It features Ackermann steering with zero-radius turning capability, enabling agile maneuvering in complex environments. The SLAM suite integrates a 32-beam LiDAR, IMU sensor, and depth compensation camera to support collision detection and realworld localization. For manipulation capabilities, we equipped the platform with a Franka Emika Research 3 robotic arm. Two Intel RealSense D435i cameras serve as primary vision sensors (one mounted at 1.5 m above ground level and another at the manipulator’s end effector), providing 848 × 480 RGB-D imagery with an 87° field of view. To mitigate depth-sensing artifacts including holes and edge discontinuities, we apply spatio-temporal and hole-filling filters, constraining the effective sensing range to 0.3–8.0 m. An industrial-grade computer equipped with an NVIDIA RTX 4090 GPU handles all real-time processing, including BSC-Nav, SLAM computation, and manipulator control, ensuring end-to-end system responsiveness.

###### Implementation details

Before task execution, BSC-Nav requires environmental perception to construct preliminary landmark memory and cognitive map representations. For simulation environments, we implement a frontier-based [44] autonomous exploration strategy for spatial memory construction: at each time step, the

system generates a height map through depth projection to identify boundaries between explored and unexplored regions. On traversable boundaries, the system selects the nearest frontier point to the current position as the next exploration target, utilizing Habitat’s scene-mesh-based greedy navigator for low-level action planning. Upon reaching each frontier point, the agent performs a 360-degree rotation to comprehensively perceive the surrounding environment. This exploration process continues until a predetermined iteration limit is reached, which is dynamically adjusted based on scene scale. The iteration count is defined as half of the traversable area.

For real-world deployment, safety constraints necessitate manual teleoperation for pre-collecting environmental observations to construct structured spatial memory. Detailed implementations of these methods are available in our code repository.

During task execution, both landmark memory and cognitive map maintain continuous updates. Parameter configurations are as follows: for landmark memory, the detector confidence threshold is set to 0.55 with a spatial overlap distance of 1.0 m; for cognitive map, we configure a voxel resolution of δ = 0.1 m3 with grid dimension G = 1000 and buffer capacity B = 10 per voxel. For each navigation episode, the system first retrieves candidate target locations through the working memory module, with maximum candidates K = 3 for landmark memory retrieval and Q = 3 for cognitive map retrieval. We employ Stable Diffusion 3.5-Medium for cognitive map retrieval, generating three images per batch to mitigate stochasticity in visual imagination. The system subsequently navigates to each candidate location following the planned exploration sequence, performing target verification upon arrival at each candidate. Task success is determined when at least one candidate passes verification. If all candidates fail verification, the task is deemed unsuccessful.

###### Baselines

End-to-end navigation methods leverage deep neural networks to directly map egocentric observations to action sequences, implicitly encoding spatialgeometric priors and semantic knowledge within network parameters. They must simultaneously learn spatial memory tracking and action planning, requiring extensive trajectory data or expert demonstrations for effective training. For example, PixNav [11], which performs greedy navigation by predicting optimal actions toward salient pixels in the current view. DAgRL [41], which integrates pre-trained vision–language encoders with historical action embeddings via transformer architectures and applies DAgger-based [74] online policy optimization. PSL [42], which uses CLIP to encode both visual observations and textual goals, minimizing semantic discrepancies between them. Although these methods have shown strong performance in simulation environments, they often struggle to generalize to the real world due to visual domain shifts. Recent Vision-Language-Action (VLA) models [75–77] address these limitations by scaling model capacity and diversifying training trajectories. Uni-NavId exemplifies this trend, achieving strong performance in parsing

complex multi-modal instructions. However, these large-scale models incur significant computational overhead, limiting their action generation frequency, and still lack mechanisms for persistent spatial memory integration.

Modular navigation methods offer a more interpretable and adaptable alternative by constructing explicit spatial representations to support the generation of low-level policies. For example, GOAT [48] projects closed set semantic segmentation onto top-down semantic maps and learns policies to predict optimal sub-goals from these maps. MOD-IIN [46] extends this paradigm to image-goal navigation. VLFM [13] employs the vision-language model, BLIP-2 [78], to construct probabilistic frontier maps indicating target likelihood at candidate viewpoints. UniGoal [47] unifies object categories, instance images, and text descriptions through abstract graph-based representations to support multi-granular spatial modeling. By decoupling low-level control from perceptual inference, these methods exhibit improved sim-to-real transferability. However, current implementations often suffer from incomplete spatial memory by modeling either landmarks or route knowledge in isolation, limiting their performance in higher-level spatial reasoning.

### Data Availability

All simulation scene data and evaluation benchmarks used in this study are publicly available through open-source platforms or datasets. The MP3D and HM3D scene data can be accessed through the Habitat-sim repository (https://github.com/facebookresearch/habitat-sim/ blob/main/DATASETS.md). Navigation benchmarks including object-goal navigation, image-instance navigation, and VLN-CE R2R datasets are available through Habitat-lab (https://github.com/facebookresearch/habitat-lab/ blob/main/DATASETS.md). The open-vocabulary object navigation benchmark and configuration files are from Yokoyama et al. [41] (https://github. com/naokiyokoyama/ovon). Text-instance navigation data and configurations are from Sun et al. [42] (https://github.com/XinyuSun/PSL-InstanceNav). Active embodied question-answering data are from the OpenEQA dataset by Majumdar et al. [54] (https://github.com/facebookresearch/open-eqa).

### Code Availability

The BSC-Nav implementation for both simulation and real-world experiments, along with data analysis and visualization scripts, is publicly available on GitHub (https://github.com/Heathcliff-saku/BSC-Nav). The repository is organized into two branches: the sim branch contains simulation implementations, benchmark configurations, and evaluation scripts; the phy branch includes low-level control interfaces adapted for our robotic platform and BSCNav deployment scripts. Researchers can refer to these implementations to deploy BSC-Nav on custom-built embodied agents.

### Acknowledgments

This work was supported by NSFC Projects (Nos. 92370124, 92248303, 62276149, 62350080, 62406160), BNRist (BNR2022RC01006).

### Author Contributions Statement

S.R., L.W. and H.S. conceived the study. S.R., L.W. and S.L. designed the computational model. S.R. and C.K. performed the main experiments, assisted by Q.Z.. S.R. and Q.Z. completed the design of the embodied robot and algorithm deployment. L.W. and S.R. drafted the paper. H.S. and X.W. supervised the project. H.S. offered funding and critically revised the paper. All authors read and approved the final version.

### Competing Interests Statement

The authors declare no competing interests.

### References

- [1] Gupta, A., Savarese, S., Ganguli, S., Fei-Fei, L.: Embodied intelligence via learning and evolution. Nature Communications 12(1), 5721 (2021)
- [2] Malanchini, M., Rimfeld, K., Shakeshaft, N.G., McMillan, A., Schofield, K.L., Rodic, M., Rossi, V., Kovas, Y., Dale, P.S., Tucker-Drob, E.M., et al.: Evidence for a unitary structure of spatial cognition beyond general intelligence. NPJ Science of Learning 5(1), 9 (2020)
- [3] Bellmund, J.L., G¨rdenfors, P., Moser, E.I., Doeller, C.F.: Navigating cognition: Spatial codes for human thinking. Science 362(6415), 6766

(2018)

- [4] Epstein, R.A., Patai, E.Z., Julian, J.B., Spiers, H.J.: The cognitive map in humans: spatial navigation and beyond. Nature neuroscience 20(11), 1504–1513 (2017)
- [5] Denis, M., Loomis, J.M.: Perspectives on human spatial cognition: memory, navigation, and environmental learning. Springer (2007)
- [6] Yang, J., Yang, S., Gupta, A.W., Han, R., Fei-Fei, L., Xie, S.: Thinking in space: How multimodal large language models see, remember, and recall spaces. In: Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 10632–10643 (2025)
- [7] Tucker, S.: A systematic review of geospatial location embedding approaches in large language models: A path to spatial ai systems. arXiv preprint arXiv:2401.10279 (2024)
- [8] Papadimitriou, F.: Spatial Artificial Intelligence. Springer
- [9] Banino, A., Barry, C., Uria, B., Blundell, C., Lillicrap, T., Mirowski, P., Pritzel, A., Chadwick, M.J., Degris, T., Modayil, J., et al.: Vectorbased navigation using grid-like representations in artificial agents. Nature 557(7705), 429–433 (2018)
- [10] Bermudez-Contreras, E., Clark, B.J., Wilber, A.: The neuroscience of spatial navigation and the relationship to artificial intelligence. Frontiers in Computational Neuroscience 14, 63 (2020)
- [11] Cai, W., Huang, S., Cheng, G., Long, Y., Gao, P., Sun, C., Dong, H.: Bridging zero-shot object navigation and foundation models through pixel-guided navigation skill. In: 2024 IEEE International Conference on Robotics and Automation (ICRA), pp. 5228–5234 (2024). IEEE
- [12] Zhang, J., Wang, K., Wang, S., Li, M., Liu, H., Wei, S., Wang, Z., Zhang, Z., Wang, H.: Uni-navid: A video-based vision-language-action model

for unifying embodied navigation tasks. arXiv preprint arXiv:2412.06224

(2024)

- [13] Yokoyama, K., et al.: Vlfm: Vision-language frontier maps for zero-shot semantic navigation. arXiv preprint arXiv:2403.08000 (2024)
- [14] Ramakrishnan, S.K., Wijmans, E., Kraehenbuehl, P., Koltun, V.: Does spatial cognition emerge in frontier models? arXiv preprint arXiv:2410.06468 (2024)
- [15] McNamara, T.P., Hardy, J.K., Hirtle, S.C.: Subjective hierarchies in spatial memory. Journal of Experimental Psychology: Learning, Memory, and Cognition 15(2), 211 (1989)
- [16] Werner, S., Krieg-Bru¨ckner, B., Mallot, H.A., Schweizer, K., Freksa, C.: Spatial cognition: The role of landmark, route, and survey knowledge in human and robot navigation. In: Informatik’97 Informatik Als Innovationsmotor: 27. Jahrestagung der Gesellschaft Fu¨r Informatik Aachen, 24.–26. September 1997, pp. 41–50 (1997). Springer
- [17] Siegel, A.W., White, S.H.: The development of spatial representations of large-scale environments. Advances in Child Development and Behavior 10, 9–55 (1975)
- [18] Chaplot, D.S., et al.: Object goal navigation using goal-oriented semantic exploration. In: NeurIPS (2020)
- [19] Yadav, A., et al.: Offline reinforcement learning for visual navigation. In: ICCV (2022)
- [20] Shah, D., Sridhar, A., Dashora, N., Stachowicz, K., Black, K., Hirose, N., Levine, S.: Vint: A foundation model for visual navigation. arXiv preprint arXiv:2306.14846 (2023)
- [21] OpenAI: GPT-4V(ision) System Card. https://cdn.openai.com/papers/ GPTV System Card.pdf (2023)

- [22] Yang, H., et al.: Qwen-vl: A versatile vision-language model. arXiv preprint arXiv:2308.12966 (2023)
- [23] Richter, K.-F., Winter, S.: Landmarks. Springer Cham Heidelberg New York Dordrecht London. doi 10(978-3), 1 (2014)
- [24] Jansen-Osmann, P., Fuchs, P.: Wayfinding behavior and spatial knowledge of adults and children in a virtual environment: The role of landmarks. Experimental Psychology 53(3), 171–181 (2006)
- [25] Wolbers, T., Weiller, C., Bu¨chel, C.: Neural foundations of emerging route

- knowledge in complex spatial environments. Cognitive Brain Research 21(3), 401–411 (2004)
- [26] Chrastil, E.R., Warren, W.H.: Active and passive spatial learning in human navigation: acquisition of survey knowledge. Journal of Experimental Psychology: Learning, Memory, and Cognition 39(5), 1520 (2013)
- [27] Baddeley, A.: Working memory: theories, models, and controversies. Annual Review of Psychology 63, 1–29 (2012)
- [28] Awh, E., Jonides, J.: Overlapping mechanisms of attention and spatial working memory. Trends in Cognitive Sciences 5(3), 119–126 (2001)
- [29] Logie, R.H.: Visuo-spatial Working Memory. Psychology Press, ???

(2014)

- [30] Oquab, M., et al.: Dinov2: Learning robust visual features without supervision. In: CVPR (2023)
- [31] Darcet, T., Oquab, M., Mairal, J., Bojanowski, P.: Vision Transformers Need Registers (2023)
- [32] Friston, K.: The free-energy principle: a unified brain theory? Nature reviews neuroscience 11(2), 127–138 (2010)
- [33] Friston, K.: The free-energy principle: a rough guide to the brain? Trends in cognitive sciences 13(7), 293–301 (2009)
- [34] OpenAI: Hello gpt-4o. https://openai.com/index/hello-gpt-4o/ (2024)
- [35] Esser, P., Kulal, S., Blattmann, A., Entezari, R., Mu¨ller, J., Saini, H., Levi, Y., Lorenz, D., Sauer, A., Boesel, F., et al.: Scaling rectified flow transformers for high-resolution image synthesis. In: Forty-first International Conference on Machine Learning (2024)
- [36] Chang, A., Dai, A., Funkhouser, T., Halber, M., Niessner, M., Savva, M., Song, S., Zeng, A., Zhang, Y.: Matterport3d: Learning from rgb-d data in indoor environments. arXiv preprint arXiv:1709.06158 (2017)
- [37] Ramakrishnan, S.K., Gokaslan, A., Wijmans, E., Maksymets, O., Clegg, A., Turner, J., Undersander, E., Galuba, W., Westbury, A., Chang, A.X., et al.: Habitat-matterport 3d dataset (hm3d): 1000 large-scale 3d environments for embodied ai. arXiv preprint arXiv:2109.08238 (2021)
- [38] Savva, M., Kadian, A., Maksymets, O., Zhao, Y., Wijmans, E., Jain, B., Straub, J., Liu, J., Koltun, V., Malik, J., et al.: Habitat: A platform for embodied ai research. In: Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 9339–9347 (2019)

- [39] Batra, D., Gokaslan, A., Kembhavi, A., Maksymets, O., Mottaghi, R., Savva, M., Toshev, A., Wijmans, E.: Objectnav revisited: On evaluation of embodied agents navigating to objects. arXiv preprint arXiv:2006.13171

(2020)

- [40] Sun, J., Wu, J., Ji, Z., Lai, Y.-K.: A survey of object goal navigation. IEEE Transactions on Automation Science and Engineering (2024)
- [41] Yokoyama, N., Ramrakhya, R., Das, A., Batra, D., Ha, S.: Hm3d-ovon: A dataset and benchmark for open-vocabulary object goal navigation. In: 2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pp. 5543–5550 (2024). IEEE
- [42] Sun, X., Liu, L., Zhi, H., Qiu, R., Liang, J.: Prioritized semantic learning for zero-shot instance navigation. In: European Conference on Computer Vision, pp. 161–178 (2024). Springer
- [43] Krantz, J., Lee, S., Malik, J., Batra, D., Chaplot, D.S.: Instance-specific image goal navigation: Training embodied agents to find object instances. arXiv preprint arXiv:2211.15876 (2022)
- [44] Gervet, T., Chintala, S., Batra, D., Malik, J., Chaplot, D.S.: Navigating to objects in the real world. Science Robotics 8(79), 6991 (2023)
- [45] Anderson, P., Chang, A., Chaplot, D.S., Dosovitskiy, A., Gupta, S., Koltun, V., Kosecka, J., Malik, J., Mottaghi, R., Savva, M., et al.: On evaluation of embodied navigation agents. arXiv preprint arXiv:1807.06757

(2018)

- [46] Krantz, J., Gervet, T., Yadav, K., Wang, A., Paxton, C., Mottaghi, R., Batra, D., Malik, J., Lee, S., Chaplot, D.S.: Navigating to objects specified by images. In: Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 10916–10925 (2023)
- [47] Yin, H., et al.: Unigoal: Towards universal zero-shot goal-oriented navigation. arXiv preprint arXiv:2503.10630 (2025)
- [48] Chang, M., Gervet, T., Khanna, M., Yenamandra, S., Shah, D., Min, S.Y., Shah, K., Paxton, C., Gupta, S., Batra, D., Mottaghi, R., Malik, J., Chaplot, D.S.: Goat: Go to any thing. arXiv preprint arXiv:2311.06430

(2023)

- [49] Anderson, P., Wu, Q., Teney, D., Bruce, J., Johnson, M., Su¨nderhauf, N., Reid, I., Gould, S., van den Hengel, A.: Vision-and-language navigation: Interpreting visually-grounded navigation instructions in real environments. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR) (2018)

- [50] Krantz, J., Wijmans, E., Majundar, A., Batra, D., Lee, S.: Beyond the nav-graph: Vision and language navigation in continuous environments. In: European Conference on Computer Vision (ECCV) (2020)
- [51] OpenAI: Introducing openai o3 and o4-mini. https://openai.com/index/introducing-o3-and-o4-mini/ (2025)
- [52] Das, A., Datta, S., Gkioxari, G., Lee, S., Parikh, D., Batra, D.: Embodied question answering. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 1–10 (2018)
- [53] Wijmans, E., Datta, S., Maksymets, O., Das, A., Gkioxari, G., Lee, S., Essa, I., Parikh, D., Batra, D.: Embodied question answering in photorealistic environments with point cloud perception. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 6659–6668 (2019)
- [54] Majumdar, A., Ajay, A., Zhang, X., Putta, P., Yenamandra, S., Henaff, M., Silwal, S., Mcvay, P., Maksymets, O., Arnaud, S., et al.: Openeqa: Embodied question answering in the era of foundation models. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 16488–16498 (2024)
- [55] Ren, A.Z., Clark, J., Dixit, A., Itkina, M., Majumdar, A., Sadigh, D.: Explore until confident: Efficient exploration for embodied question answering. arXiv preprint arXiv:2403.15941 (2024)
- [56] Kim, M.J., Pertsch, K., Karamcheti, S., Xiao, T., Balakrishna, A., Nair, S., Rafailov, R., Foster, E., Lam, G., Sanketi, P., et al.: Openvla: An opensource vision-language-action model. arXiv preprint arXiv:2406.09246

(2024)

- [57] Liu, S., Wu, L., Li, B., Tan, H., Chen, H., Wang, Z., Xu, K., Su, H., Zhu, J.: Rdt-1b: a diffusion foundation model for bimanual manipulation. arXiv preprint arXiv:2410.07864 (2024)
- [58] Black, K., Brown, N., Driess, D., Esmail, A., Equi, M., Finn, C., Fusai, N., Groom, L., Hausman, K., Ichter, B., et al.: \pi 0: A vision-language-action flow model for general robot control. arXiv preprint arXiv:2410.24164

(2024)

- [59] Wu, Z., Wang, Z., Xu, X., Lu, J., Yan, H.: Embodied task planning with large language models. arXiv preprint arXiv:2307.01848 (2023)
- [60] Hilton, C., Wiener, J.: Route sequence knowledge supports the formation of cognitive maps. Hippocampus 33(11), 1161–1170 (2023)

- [61] Zador, A., Escola, S., Richards, B., Olveczky,¨ B., Bengio, Y., Boahen, K., Botvinick, M., Chklovskii, D., Churchland, A., Clopath, C., et al.: Catalyzing next-generation artificial intelligence through neuroai. Nature Communications 14(1), 1597 (2023)
- [62] Zou, Q., Sun, Q., Chen, L., Nie, B., Li, Q.: A comparative analysis of lidar slam-based indoor navigation for autonomous vehicles. IEEE Transactions on Intelligent Transportation Systems 23(7), 6907–6921 (2021)
- [63] Cheng, T., Song, L., Ge, Y., Liu, W., Wang, X., Shan, Y.: Yoloworld: Real-time open-vocabulary object detection. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 16901–16911 (2024)
- [64] Jatavallabhula, K.M., Kuwajerwala, A., Gu, Q., Omama, M., Chen, T., Maalouf, A., Li, S., Iyer, G., Saryazdi, S., Keetha, N., et al.: Conceptfusion: Open-set multimodal 3d mapping. arXiv preprint arXiv:2302.07241

(2023)

- [65] Huang, C., Mees, O., Zeng, A., Burgard, W.: Visual language maps for robot navigation. In: 2023 IEEE International Conference on Robotics and Automation (ICRA), pp. 10608–10615 (2023). IEEE
- [66] Rombach, R., Blattmann, A., Lorenz, D., Esser, P., Ommer, B.: Highresolution image synthesis with latent diffusion models. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 10684–10695 (2022)
- [67] Kriegel, H.-P., Kr¨ger, P., Sander, J., Zimek, A.: Density-based clustering. Wiley interdisciplinary reviews: data mining and knowledge discovery 1(3), 231–240 (2011)
- [68] Hart, P.E., Nilsson, N.J., Raphael, B.: A formal basis for the heuristic determination of minimum cost paths. IEEE transactions on Systems Science and Cybernetics 4(2), 100–107 (1968)
- [69] Ro¨smann, C., Feiten, W., W¨sch, T., Hoffmann, F., Bertram, T.: Trajectory modification considering dynamic constraints of autonomous robots. In: ROBOTIK 2012; 7th German Conference on Robotics, pp. 1–6 (2012). VDE
- [70] Ro¨smann, C., Feiten, W., W¨sch, T., Hoffmann, F., Bertram, T.: Efficient trajectory optimization using a sparse model. In: 2013 European Conference on Mobile Robots, pp. 138–143 (2013). IEEE
- [71] Puig, X., Undersander, E., Szot, A., Cote, M.D., Yang, T.-Y., Partsey, R.,

- Desai, R., Clegg, A.W., Hlavac, M., Min, S.Y., et al.: Habitat 3.0: A cohabitat for humans, avatars and robots. arXiv preprint arXiv:2310.13724 (2023)
- [72] Yadav, K., Krantz, J., Ramrakhya, R., Ramakrishnan, S.K., Yang, J., Wang, A., Turner, J., Gokaslan, A., Berges, V.-P., Mootaghi, R., Maksymets, O., Chang, A.X., Savva, M., Clegg, A., Chaplot, D.S., Batra, D.: Habitat Challenge 2023. https://aihabitat.org/challenge/2023/

(2023)

- [73] Wang, W., Lv, Q., Yu, W., Hong, W., Qi, J., Wang, Y., Ji, J., Yang, Z., Zhao, L., XiXuan, S., et al.: Cogvlm: Visual expert for pretrained language models. Advances in Neural Information Processing Systems 37, 121475–121499 (2024)
- [74] Ross, S., Gordon, G., Bagnell, D.: A reduction of imitation learning and structured prediction to no-regret online learning. In: Proceedings of the Fourteenth International Conference on Artificial Intelligence and Statistics, pp. 627–635 (2011). JMLR Workshop and Conference Proceedings
- [75] Zhang, J., Wang, K., Xu, R., Zhou, G., Hong, Y., Fang, X., Wu, Q., Zhang, Z., Wang, H.: Navid: Video-based vlm plans the next step for vision-and-language navigation. arXiv preprint arXiv:2402.15852 (2024)
- [76] Zhang, J., et al.: Uni-navid: A video-based vision-language-action model for unifying embodied navigation tasks. arXiv preprint arXiv:2412.06224

(2024)

- [77] Cheng, A.-C., Ji, Y., Yang, Z., Gongye, Z., Zou, X., Kautz, J., Bıyık, E., Yin, H., Liu, S., Wang, X.: Navila: Legged robot vision-language-action model for navigation. arXiv preprint arXiv:2412.04453 (2024)
- [78] Li, J., Li, D., Savarese, S., Hoi, S.: Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In: International Conference on Machine Learning, pp. 19730–19742 (2023). PMLR

### A Prompt Template

BSC-Nav deeply integrates the perceptual and reasoning capabilities of Multimodal Large Language Models (MLLMs) across multiple critical system components, leveraging their strengths in visual understanding, semantic reasoning, and task planning. To fully harness the potential of MLLMs and ensure their reliability in various tasks, we develop a comprehensive prompt engineering encompassing core functional modules including target verification, instruction decomposition, etc. Below provide core prompt template for each functional module. For specific model versions, and complete prompt implementations, please kindly refer to our code repository.

|Prompt for Landmark Memory Retrieval<br><br>Given a textual description of a navigation target and a landmark memory list containing instances of detected objects, each with a label, a concise description, a 3D location (three numerical coordinates), and a confidence score, you need to determine the most suitable memory instance to fulfill the navigation request.<br><br>1. Landmark memory data structure: The memory is a list of objects in the environment, where each object is represented as a JSON-like structure of the following form. {<br><br>label : <string >, description : <string >, loc : [< float >, <float >, <float >] , confidence : <float ><br><br>}<br><br>2. Your task:<br><br><br>2.1 Understand the target description: You will be given a textual goal description of a navigation target, such as “A marble island in a kitchen.” Your first step is to interpret this description and deduce which object label or description from the memory best matches it semantically.<br>2.2 Identify the relevant instances: Once you have determined the best matching label, filter the memory list to only those instances whose label matches (or closely matches) that label.<br>2.3 Evaluate confidence and consolidate duplicates: Among these filtered instances, consider that multiple memory entries may actually represent the same object, possibly due to partial overlaps or multiple detections.<br><br><br>1) Look at their loc coordinates. If multiple instances with the same label have very close or nearly identical coordinates, treat them as the same object. 2) Determine which set of coordinates (if there are multiple distinct sets) is the most reliable representation of the object. Reliability is judged primarily by the highest confidence value. If multiple instances|
|---|

|cluster together with similar locations, select the one with the highest confidence or, if confidence is similar, the one that best aligns with the object as described.<br><br>2.4 Select the final loc: After you have grouped instances and decided which group best represents the target object, output the coordinates (loc) of the best match. If multiple objects (≥3 items) match the description equally well, choose the three coordinates (loc) with the highest confidence.<br>2.5 Produce a final answer: Return the selected location coordinates as the final answer, must be in the format {Nav Loc 1: [...], Nav Loc 2: [...], Nav Loc 3: [...]} {Nav Loc: Unable to find}<br><br><br>3. Important details<br><br>3.1 Always provide reasoning internally (you may do it in hidden scratchpads if available) before giving the final result. The final user-visible answer should be concise and directly address the task.<br>3.2 If no objects are found that are semantically relevant to the target description, explicitly indicate that no suitable object was found.<br>3.3 Follow these steps for every input you receive. Navigation target: {text prompt} Memory: {landmark memory} Now please start thinking one step at a time and then Briefly tell me the target locations.<br><br><br>|
|---|

|Prompt Template for Association-Enhanced Text Generation<br><br>You are an expert in generating prompts for text-to-image models. Your task is to enhance a given original goal description, which often only mentions a general category or a simple phrase describing the target object, by incorporating detailed context from four current scene images.<br><br>1. Task Overview: Create a more imaginative and contextually enriched description, using elements observed in the scene to create a coherent and vivid visual. This description will guide a text-to-image model to generate an image that aligns with the style and context of the current scene. The target object must remain the primary visual focus.<br>2. Suggested Steps:<br><br><br>2.1 Understand the Environment: Extract and comprehend details from the provided observation images, such as the overall style, decoration, and elements of the scene. Consider whether this is a modern home, classical residence, exhibition hall, or office.<br>2.2 Expand the Original Description: Based on the scene analysis, enrich the original description with finer details including materials, colors, textures, placement, and environmental elements.<br>|
|---|

|2.3 Maintain Visual Focus: Ensure additional context or background details do not overshadow the main target object. The primary subject should remain the focal point, using language that emphasizes its prominence.<br>3. Guidelines for Creating Enhanced Descriptions: 3.1 Details: Include sensory details like colors, textures, lighting, and reflections.<br><br>3.2 Background Elements: Add appropriate background elements that complement the scene without detracting from focus.<br>3.3 Focus Phrasing: Use language that naturally draws attention to the target object (e.g., “centered”, “prominently placed”, “as the focal point”).<br>3.4 Balance: Strike a balance between richness and simplicity. The target object should always dominate the final image.<br><br><br>4. Output Requirements:<br><br><br>4.1 Provide a refined and detailed description in English.<br>4.2 Ensure the enhancement creates a vivid, coherent, and engaging scene.<br>4.3 Avoid overly complex narratives or distracting elements.<br>4.4 Keep the description concise, limiting it to 70 words or less.<br>5. Examples:<br><br><br>Example 1:<br><br>• Original: A green vase.<br>• Enhanced: A vibrant green ceramic vase, with a glossy, smooth surface, placed centrally on a polished wooden table. Soft natural light illuminates the vase from the large window behind it, casting shadows on the table. The surrounding room is decorated in minimalist modern style with neutral tones, ensuring the vase is the central focal point.<br><br><br>Example 2:<br><br><br>• Original: A armchair.<br>• Enhanced: A sleek, modern blue armchair, upholstered in soft velvet, positioned prominently in a stylish living room. The chair is placed near a large floor-to-ceiling window, allowing natural light to highlight its deep blue hue. The room features minimalist decor with white walls, light wood flooring, and a few abstract art pieces on the walls.<br><br><br>Original: {text prompt}, Observations: {[img list]}. Please follow the above requirements and examples to enhance this description, think step by step and give your analysis process and the final enhancement description following the format: analysis process: [your analysis process here] enhancement description: [your enhancement description here]<br><br>|
|---|

|Prompt for Goal Verification and Affordance<br><br>You will be provided with a navigation observation image from a robot, and a textual description of the navigation goal. Please follow the steps below to determine whether the current navigation task is successful.<br><br>1. Determine Target Presence: Analyze the provided images to ascertain whether the navigation goal is present in these images AND close enough (within 2 meters). This means evaluating if the robot has arrived near the target location. Be careful not to misclassify similar categories (e.g. sofas and chairs are easily confused).<br>2. Determine whether need to move forward: If you have found the target according to step 1, you need to further determine whether you need to move forward a small step to get closer to the target object. If need to, answer “need forward: yes”. If you think you are close enough (within 1m), answer “need forward: no”.<br>3. Output Format:<br><br>3.1 First Line: Success: yes OR Success: no<br><br>3.2 Second Line (only when “success: yes”): need forward: yes OR need forward: no<br><br>3.3 Third Line: Give your analysis results in detail<br>4. Important Notes:<br><br><br>4.1 Please analyze according to the above requirements and respond strictly in the specified format. 4.2 Be precise in distance estimation and target identification. target description: {text prompt} observation images: {img} Now please start thinking step by step.<br><br><br>|
|---|

|Prompt for Complex Instruction Decompose<br><br>You will get a text instruction for long-distance navigation in an indoor environment. Now, your task is to decompose the text instruction into reasonable and clear sub-task goals to help the agent complete the complex navigation task step by step.<br><br>1. Task Description: All sub-tasks need to be expressed in the form of {move to ...}, where the target in {...} can be a word or description of an object, or a description of a room area.<br><br>2. Examples: Example 1:<br><br><br>• Text prompt: ”Walk into the hallway and at the top of the stairs turn left and walk into the bedroom. Walk past the right side of the bed|
|---|

|and turn right into the closet and all the way through to the bathroom. Stop in front of the toilet in the bathroom.”<br><br>• Response:<br><br>1. Move to the {stairs at the end of the hallway}<br>2. Move to the {bed in the bedroom}<br>3. Move to the {closet}<br>4. Move to the {toilet in the bathroom}<br><br><br>Example 2:<br><br>• Text prompt: “Go to the wooden stairs. Go up the stairs and go between the couch and the table. Walk into the house through the sliding glass door. Go to the television. Go to the refrigerator. Go to the front of the toaster and stop”<br>• Response:<br><br><br>1. Move to the {wooden stairs}<br>2. Move to the {area between a couch and a table}<br>3. Move to the {sliding glass door}<br>4. Move to the {television}<br>5. Move to the {refrigerator}<br>6. Move to the {toaster}<br><br><br>3. Output Requirements:<br><br>3.1 Please planning the following text prompt into sub-goals and respond strictly in the specified format.<br>3.2 Do not include any other information. Text prompt: {text prompt}<br><br><br>|
|---|

|Prompt for Determining EQA Task’s Waypoints<br><br>You will act as an agent to complete the task of assisting scene embodied question answering. I will provide you with a question about the scene. In order to answer this question, we must first navigate to the vicinity of the instance involved in the question.<br><br>1. Task Description: Now, your task is to analyze and determine the description of the target instance you need to move to based on the current question, which can include some necessary spatial context, such as what type of room this target instance is in and what clear objects exist around it.<br>2. Special Case: If you think it is difficult to infer the exact target instance for the type of question provided, please output “We need to go around and check”<br><br>3. Output Requirements:<br>|
|---|

|You need to output the description directly without including other analysis and additional response content.|
|---|

### B Supplementary Figures and Videos

###### Supplementary Video Descriptions

- Video 1. Navigation demonstrations of BSC-Nav in simulation environments, showcasing nine successful episodes across object-goal navigation, open-vocabulary navigation, text-instance navigation, and image-instance navigation tasks. Each episode displays egocentric observations, top-down maps, and goal descriptions. In the top-down view, blue squares indicate target locations, with surrounding blue regions representing success zones (within a one-meter radius). These demonstrations illustrate BSC-Nav’s efficient general-purpose navigation capabilities.
- Video 2. Long-horizon instruction-following navigation of BSC-Nav in simulation environments, comprising six successful episodes from the VLN-CE R2R benchmark with different instruction complexity. Each episode includes egocentric observations, top-down maps, and human instructions. Blue squares mark target locations, while red squares indicate necessary intermediate waypoints along the instructed path.
- Video 3. Object-goal navigation using BSC-Nav in real-world environments: navigating to a table.
- Video 4. Text-instance navigation using BSC-Nav in real-world environments: navigating to a circular sofa with a small tree in the center.
- Video 5. Text-instance navigation using BSC-Nav in real-world environments: navigating to a lounge area with a soft orange cushion.
- Video 6. Image-instance navigation using BSC-Nav in real-world environments: navigating to an access door (image-specified).
- Video 7.Image-instance navigation using BSC-Nav in real-world environments: navigating to a key model (image-specified).
- Video 8. Mobile manipulation using BSC-Nav in real-world environments,

- Case 1: clearing a desktop, involving one navigation target.

Video 9. Mobile manipulation using BSC-Nav in real-world environments,

- Case 2: relocating a cookie box, involving two navigation targets.

Video 10. Mobile manipulation using BSC-Nav in real-world environments,

- Case 3: preparing oatmeal breakfast, involving three navigation targets.

[Figure 602]

[Figure 603]

[Figure 604]

[Figure 605]

[Figure 606]

[Figure 607]

Frontier-based exploration

[Figure 608]

[Figure 609]

[Figure 610]

Observations

|[Figure 611]<br><br>[Figure 612]|
|---|

|[Figure 613]<br><br>[Figure 614]|
|---|

|[Figure 615]<br><br>[Figure 616]|
|---|

|[Figure 617]<br><br>[Figure 618]|
|---|

[Figure 619]

[Figure 620]

[Figure 621]

[Figure 622]

|[Figure 623]<br><br>[Figure 624]|
|---|

[Figure 625]

[Figure 626]

[Figure 627]

[Figure 628]

[Figure 629]

[Figure 630]

[Figure 631]

YOLO-World DINO-V2

[Figure 632]

[Figure 633]

|[Figure 634]<br><br>[Figure 635]|
|---|

[Figure 636]

[Figure 637]

[Figure 638]

[Figure 639]

[Figure 640]

[Figure 641]

[Figure 642]

[Figure 643]

[Figure 644]

|Pa|tch|oken|s|
|---|---|---|---|
| | | | |
|[Figure 645]<br><br>[Figure 646]<br><br>[Figure 647]| | | |
| | | | |

[Figure 648]

[Figure 649]

Detection Info.

[Figure 650]

|[Figure 651]<br><br>[Figure 652]|
|---|

[Figure 653]

[Figure 654]

[Figure 655]

[Figure 656]

[Figure 657]

Projection Projection

[Figure 658]

[Figure 659]

[Figure 660]

[Figure 661]

[Figure 662]

Landmark Memory Cognitive Map

[Figure 663]

[Figure 664]

[Figure 665]

[Figure 666]

[Figure 667]

[Figure 668]

Working Memory

|[Figure 669]<br><br>[Figure 670]<br><br>[Figure 671]|
|---|

|“ The matching location is […], […], […], … "| |
|---|---|
|Outputs| |
| | |

###### Cognitive Map

Landmark Memory

Voxel Features

[Figure 672]

|Text-goal| |
|---|---|
| | |

…

Candidate Coords.

LLM (GPT-4)

Goal Features

Pooling

Similarity Weighted Clustering

|Image-goal|
|---|

| | |
|---|---|
| | |

|[Figure 673]<br><br>[Figure 674]<br><br>[Figure 675]|
|---|

!

Visual Imagination

[Figure 676]

[Figure 677]

[Figure 678]

[Figure 679]

[Figure 680]

|Fine-grained Description| |
|---|---|
| |T2I Model|

DINO V2

|Text-goal| |
|---|---|
| | |

Candidate Coords.

Top-N Coords.

LLM

- Fig. 1 Construction and exploitation of structured spatial memory. BSC-Nav first initializes structured spatial memory through frontier-based exploration strategies. Observational information is processed in parallel, employing YOLO-World for salient object detection and DINO-V2 for patch-level visual embedding extraction, which are respectively used to construct the landmark memory and the cognitive map. During navigation initialization, the working memory module performs hierarchical retrieval based on instruction modality and granularity. For simple textual targets, BSC-Nav employs LLMs for contextual retrieval over the landmark memory. For specific textual descriptions and image targets, BSC-Nav performs associative enhanced retrieval over the cognitive map to generate candidate exploration sequences.

- a
- b
- c d

HM3D

###### MP3D

[Figure 681]

[Figure 682]

[Figure 683]

[Figure 684]

[Figure 685]

[Figure 686]

[Figure 687]

[Figure 688]

[Figure 689]

[Figure 690]

[Figure 691]

[Figure 692]

[Figure 693]

[Figure 694]

[Figure 695]

[Figure 696]

[Figure 697]

[Figure 698]

[Figure 699]

[Figure 700]

[Figure 701]

[Figure 702]

[Figure 703]

[Figure 704]

[Figure 705]

[Figure 706]

[Figure 707]

[Figure 708]

[Figure 709]

[Figure 710]

[Figure 711]

[Figure 712]

- (a) UniGoal
- (b) BSC-Nav (W/o Cog. map)
- (c) BSC-Nav (W/o Landmark)
- (d) BSC-Nav (Full)

80

60

60

60

61.9

78.5

75.8

57.6

57.4

56.5

60

49.4

62.8

47.7

###### SPL(%)

SPL(%)

###### SR(%)

###### SR(%)

40

40

40

54.5

41.0

38.8

40

30.2

29.0

25.1

20

20

20

20

SR (%)

16.4

SPL (%)

0

0

0

0

a b c d

a b c d

a b c d

a b c d

MP3D HM3D

100

100

91.3

SR (%)

99.2

85.4

84.7

75.3

80

80

SPL (%)

72.1

69.9

82.1

67.2

66.1

64.5

63.6

73.1

73.0

72.0

59.4

57.7

57.7

57.2

69.7

56.9

69.0

55.0

60

60

66.0

53.1

SR (%)

50.9

50.5

50.4

61.5

47.4

57.8

44.7

44.4

42.2

41.3

41.2

52.5

37.1

34.9

46.9

34.4

40

40

SPL (%)

28.9

25.0

24.4

19.8

16.1

20

20

11.2

5.3

0.0

0.0

0.0

0.0

0.0

0.0

0

0

bedchairplantsofatoilettvmonitor bedchairplantsofatoilettvmonitor

bathtubcabinetbedchairchestclothescountercushionfireplacegympictureplantseatingshowersinksofastooltabletoilettvmonitortowelbathtubcabinetbedchairchestclothescountercushionfireplacegympictureplantseatingshowersinksofastooltabletoilettvmonitortowel

###### MP3D HM3D

100

80

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

|O|N<br><br>| | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

OGN OV TIN IIN

OGN

- 0
- 1
- 2
- 3
- 4
- 5
- 6

- 0
- 1
- 2
- 3
- 4
- 5
- 6

CumulativeSPL(%)

| |[Figure 713]<br><br>[Figure 714]<br><br>[Figure 715]<br><br>[Figure 716]<br><br>[Figure 717]<br><br>[Figure 718]<br><br>[Figure 719]<br><br>[Figure 720]| | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

CumulativeSR(%)

OVON

80

| |[Figure 721]<br><br>[Figure 722]<br><br>[Figure 723]<br><br>[Figure 724]<br><br>[Figure 725]<br><br>[Figure 726]<br><br>[Figure 727]<br><br>[Figure 728]| | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

60

TIN

60

IIN

40

40

20

20

0

0

0 1 2 3 4 5 6

0 1 2 3 4 5 6

Num. of explored candidates

Num. of explored candidates

B Q B+Q

Q B B+Q

- Fig. 2 Additional results of navigation performance. a, Comparison of SR and SPL between BSC-Nav and the state-of-the-art Unigoal method in object-goal navigation using landmark memory only, cognitive map only, and both. b, Category-wise SR and SPL across 20 object categories in MP3D and 6 object categories in HM3D. c, Number of candidate coordinates explored during successful navigation episodes on MP3D and HM3D. B, landmark memory retrieval only. Q, cognitive map retrieval only. B+Q, both B and Q. It can be seen that most successful navigation episodes achieve the goal at the first explored coordinate. d, The cumulative SR and cumulative SPL as the number of explored candidate locations increases. Additional explorations improve SR (navigation success) but reduce SPL (navigation efficiency).

