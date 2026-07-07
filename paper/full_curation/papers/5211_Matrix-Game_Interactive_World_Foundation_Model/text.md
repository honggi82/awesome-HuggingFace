arXiv:2506.18701v1[cs.CV]23Jun2025

# Matrix-Game: Interactive World Foundation Model

### Yifan Zhang∗ Chunli Peng∗ Boyang Wang∗† Puyi Wang Qingcheng Zhu Fei Kang Biao Jiang Zedong Gao Eric Li Yang Liu Yahui Zhou

Skywork AI Project page: https://matrix-game-homepage.github.io

## Abstract

We introduce Matrix-Game, an interactive world foundation model for controllable game world generation. Matrix-Game is trained using a two-stage pipeline that first performs large-scale unlabeled pretraining for environment understanding, followed by action-labeled training for interactive video generation. To support this, we curate Matrix-Game-MC, a comprehensive Minecraft dataset comprising over 2,700 hours of unlabeled gameplay video clips and over 1,000 hours of high-quality labeled clips with fine-grained keyboard and mouse action annotations. Our model adopts a controllable image-to-world generation paradigm, conditioned on a reference image, motion context, and user actions. With over 17 billion parameters, Matrix-Game enables precise control over character actions and camera movements, while maintaining high visual quality and temporal coherence. To evaluate performance, we develop GameWorld Score, a unified benchmark measuring visual quality, temporal quality, action controllability, and physical rule understanding for Minecraft world generation. Extensive experiments show that Matrix-Game consistently outperforms prior open-source Minecraft world models—including Oasis and MineWorld—across all metrics, with particularly strong gains in controllability and physical consistency. Double-blind human evaluations further confirm the superiority of Matrix-Game, highlighting its ability to generate perceptually realistic and precisely controllable videos across diverse game scenarios. To facilitate future research on interactive image-to-world generation, we will open-source the Matrix-Game model weights and the GameWorld Score benchmark at https://github.com/SkyworkAI/Matrix-Game.

## 1 Introduction

World models [20,74] are fundamental to intelligent agents, enabling them to perceive, simulate, and reason about the dynamics of their environments. By internalizing the structure and behavior of the external world, these models support a wide spectrum of downstream tasks, including autonomous driving [16,55], embodied intelligence [19,21], and generative game engines [2,18]. Effective world models empower agents not only to interpret observations but also to anticipate outcomes and plan actions in open-ended, uncertain environments [27,35,37].

In recent years, video diffusion models have emerged as a leading paradigm for world modeling [1, 40,41,62,63]. Their ability to learn fine-grained spatial-temporal dynamics and generate visually coherent videos makes them particularly suited for simulating evolving worlds. Moreover, when conditioned on structured inputs such as keyboard actions and camera movements, video diffusion models provide a flexible and scalable approach to modeling complex agent-environment interactions with high fidelity [14,18,41].

∗Equal contribution. †Project Lead.

Technical Report.

[Figure 1]

- (a) Desert

[Figure 2]

- (b) Beach

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

4,-)en

Cl.)

###### II

###### 0 ' •

[Figure 7]

[Figure 8]

t_

###### Attack

(c) Forest

[Figure 9]

- (d) Hills

[Figure 10]

- (e) Icy

[Figure 11]

- (f) Plain

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

(])

[Figure 16]

[Figure 17]

w

###### s Space Attack

Attack Space Attack D Space Attack A s D Space Attack A

[Figure 18]

o,--,,

- (g) River

[Figure 19]

[Figure 20]

[Figure 21]

- (h) Mushroom

- Figure 1: Controllable world generation results of Matrix-Game across distinct Minecraft scenarios. These demos illustrate the model’s ability to handle diverse environments, ranging from desert, beach, and forest to more challenging settings like mushroom and icy biomes, while accurately responding to user control signals.

Scenario Consistency

| |Object|
|---|---|
|0|Image<br><br>20 40 60 80 100|

Mouse Accuracy

Consistency

Keyboard Accuracy

Aesthetic

Motion Smoothness

Quality

Temporal Consistency

Oasis MineWorld Ours

- Figure 2: Model performance under our GameWorld Score benchmark, covering 8 key dimensions: Image Quality, Aesthetic (scaled ×2 for visualization), Temporal Consistency, Motion Smoothness, Keyboard Accuracy, Mouse Accuracy, Object Consistency and Scenario Consistency. Our method outperforms Oasis [9] and MineWorld [18] in all aspects, particularly in controllability (keyboard and mouse accuracy) and physical consistency, while maintaining high visual and temporal quality.

While video-based world models present a promising direction for simulating and understanding dynamic worlds, they face several significant challenges. First, acquiring high-quality training data is non-trivial. Interactive video datasets with rich annotations (e.g., precise actions, camera movement) are scarce and expensive to collect, especially at scale. Second, modeling the physical dynamics of the world and achieving fine-grained controllability over time remains difficult. World models need not only to generate visually coherent videos but also to respond accurately to structured control signals. Third, the lack of a standardized evaluation benchmark makes it hard to objectively compare models or quantify improvements, limiting the research in this area.

To address the above challenges, we introduce Matrix-Game, an interactive world foundation model designed for game world generation. Our approach comprises three core components. First, we construct Matrix-Game-MC, a large-scale dataset specifically tailored for Minecraft world modeling, comprising both unlabeled gameplay video clips and richly annotated action-labeled video data. This dataset enables the learning of complex environment dynamics and interaction patterns at scale. Second, we develop Matrix-Game, a diffusion-based image-to-world generation model that supports interactive video generation conditioned on user inputs such as keyboard commands and mouse-driven camera movements. This architecture emphasizes controllability, temporal coherence, and visual fidelity in dynamic world environments. Third, to systematically evaluate world models, we develop GameWorld Score, a unified benchmark encompassing multiple evaluation dimensions (including visual quality, temporal quality, controllability, and physical rule understanding), offering a comprehensive framework for quantitative assessment of Minecraft world models.

To train Matrix-Game, we develop a two-stage strategy: an initial stage of unlabeled training for game world understanding, followed by action-labeled training for interactive world generation. Building on this training pipeline, we scale up both the model capacity (to 17B parameters) and the training data. As a result, Matrix-Game is able to generate visually coherent scenes that reflect realistic motion patterns, respond accurately to user controls (including keyboard movement and mouse-driven camera actions) and maintain high temporal consistency throughout the video. As shown in Figure 2 and Table 2, it consistently outperforms leading open-source Minecraft world models such as Oasis [9] and MineWorld [18], with particularly strong gains in action controllability and physical rule understanding. Furthermore, as illustrated in Figure 1, Matrix-Game produces high-quality, controllable videos that align with in-game physics across a variety of Minecraft scenarios. By integrating visual perception with fine-grained user control, Matrix-Game redefines video generation as an interactive process of exploration and creation, empowering users to observe, guide, and construct coherent virtual worlds from a single reference image.

## 2 Related Work

### 2.1 Video Diffusion Models

With the rapid advancement of diffusion models [23,47,48], visual content generation has witnessed remarkable progress across both image [8,34,43,45,69,72] and video domains [5,33,51,65,73]. Notably, video generation has evolved beyond traditional U-Net-based backbones, transitioning toward Transformer-based architectures [42,53] that offer enhanced scalability and temporal modeling capabilities. This shift has empowered modern video diffusion models to synthesize high-quality, temporally coherent videos of significantly longer durations [40].

The impressive fidelity has sparked growing interest in using video diffusion models to simulate complex world environments by implicitly learning physical laws, object dynamics, and causal interactions from raw video data [1,40,41,62,63]. As a result, video diffusion models are increasingly viewed as promising world models for embodied agents and interactive systems, where reasoning about physical causality, spatial continuity, and controllable dynamics is essential for tasks such as planning, navigation, and decision-making.

### 2.2 Controllable Video Generation

While text descriptions serve as the primary input for text-to-video generation [29,54], they often lack sufficient precision to capture complex spatial-temporal details, leading to ambiguous or underspecified outputs. To overcome this limitation, recent research has introduced additional control modalities to better align generation with user intent and improve content determinism. Among these, reference image guidance has proven effective for enhancing both visual fidelity and temporal consistency by anchoring the generated video to a concrete visual context [5,39].

Beyond those static signals as conditions, Direct-a-Video [64] introduces a camera embedder to influence perspective during generation, but its capabilities are constrained to only three coarse attributes (e.g., yaw/pitch/zoom). CameraCtrl [22] and MotionCtrl [56] go further by introducing finegrained trajectory-level camera guidance, offering temporally-continuous control over position and orientation. These methods enable dynamic scene exploration and user-driven cinematic effects. In parallel, World-Model-based methods [24,59] emphasize action-conditioned rollout to model physical dynamics. However, most of these approaches are limited by low visual quality or simplified action spaces. In our work, we focus on interactive image-to-world generation, where generation is guided by keyboard and mouse movement actions, enabling precise and intuitive control in open-ended environments.

### 2.3 Game Video Generation

Given the strong modeling capabilities of video diffusion models, an increasing number of studies have begun exploring their applications in game world generation [2,6,7,9,14,18,41,41,52,59,61,66,67]. Genie [6] proposes a foundation model for playable environments built upon video generation. Other methods such as DIAMOND [2], GameNGen [52], OASIS [9], and PlayGen [61] leverage diffusionbased generative models to simulate game worlds. GameGenX [7] further introduces OGameData, a dataset designed to enable game video generation and controllability. Despite their contributions, many of these approaches, including concurrent works such as GameGenX [7], OASIS [9], and WorldMem [59], tend to overfit to specific game datasets, exhibiting limited generalization across diverse game scenarios and interaction dynamics.

Some recent research works, such as Matrix [14], Genie 2 [41], GameFactory [67], and MineWorld [18], have attempted to improve control generalization in virtual environments. However, these approaches remain constrained by relatively limited model capacity and dataset scale, making it difficult to effectively capture physical rules and enable accurate interactions across diverse virtual game worlds. In this work, we aim to address these limitations by scaling up both the model size and training data, aiming to advance controllable game world generation with stronger generalization and physical rule understanding. Through this effort, we aim to push the frontier of interactive image-toworld generation, enabling more reliable and versatile agent-environment simulation grounded in user interaction.

## 3 Matrix-Game-MC: A Large-scale Dataset for Game World Generation

Our goal is to develop a world foundation model that can internalize physical dynamics, semantic structures, and support interactive video generation. To achieve this, large-scale, high-quality data is indispensable. We adopt Minecraft as the primary environment due to its diverse biomes, rich agent-environment interactions, and open-ended gameplay, which make it well-suited for learning world modeling. However, acquiring action-labeled Minecraft data through manual gameplay is both time-consuming and resource-intensive. To address this, we supplement training with a large volume of unlabeled gameplay videos to help the model learn motion dynamics and environmental rules. In parallel, we construct an automated pipeline for generating fine-grained, action-labeled video clips in a scalable manner, enabling controllable model training across varied scenarios.

### 3.1 Unlabeled Data Collection

Unlabeled data acquisition protocol. The unlabeled training dataset was systematically collected from video resources provided in the MineDojo Dataset [12]. We retrieved approximately 6,000 hours of raw gameplay footage through the dataset’s official video repositories, which included tutorial content demonstrating core game mechanics, unstructured gameplay recordings, and demonstrations of environmental interactions.

This diverse collection spans multiple biomes, including forest, desert, and snow terrain ecosystems, offering broad visual and physical coverage of the Minecraft environment. To prepare the data for training, we first employ TransNet V2 [49] to detect scene transitions and segment raw gameplay videos into single-shot clips. Segmentation is performed using FFmpeg [15] at the detected transition boundaries. Prior to processing, all video data is converted to the libx264 encoding format for compatibility and efficiency. To mitigate artifacts from gradual transitions or unstable camera motion, we discard the first and last four frames of each segmented clip. We then introduce a hierarchical filtering framework (illustrated in Figure 3) designed to curate high-quality, informative clips from raw gameplay footage. This framework integrates multiple filter modules, each capturing a different aspect of video quality, content, or temporal structure.

Unlabeled data filtering pipeline. As shown in Figure 3, our data filtering pipeline consists of three sequential stages. The first stage focuses on video quality filtering and aesthetic filtering. The second stage applies menu-state filtering, subtitle filtering, and face filtering to remove non-informative or distracting content. The final stage involves motion analysis and camera movement filtering to ensure dynamic yet visually stable clips suitable for model training. We provide detailed descriptions of each filter below.

Video quality filtering. We use DOVER [58] to assess video quality, applying genre-specific thresholds to accommodate stylistic diversity across game types. This ensures the retention of videos with sufficient resolution, clarity, and coherence for reliable model training.

Aesthetic filtering. We compute aesthetic scores using the LAION predictor [31], averaging scores across sampled frames per clip. To account for stylistic diversity across game genres, we apply adaptive, genre-aware thresholds. This ensures selected videos maintain visually coherent and appealing composition, supporting realistic generation.

Menu-State filtering. We use an Inverse Dynamics Model (IDM) [4] to detect frames without player input (such as menus, idle states, or loading screens) and exclude them. This ensures the dataset focuses on active gameplay, enhancing the model’s ability to learn action-conditioned dynamics and controllable temporal transitions.

Subtitle filtering: We apply the CRAFT text detector [3] to identify and remove videos with intrusive subtitles, stream banners, or watermarks. By focusing detection on lower-screen and high-risk regions, we preserve in-game text while excluding distracting post-production overlays, ensuring clean visual inputs for training.

Human face filtering. To ensure dataset focus on in-game environments, we use DeepFace [46] to detect and filter videos containing streamer face cams or human overlays. By checking for recurring faces in common webcam regions across multiple frames, we eliminate non-game human content, preserving scene purity and preventing the model from learning spurious visual cues.

High-quality 720p video clips (870 hours)

Motion filter

720p video clips (2,700 hours)

Raw video clips (6,000 hours)

Camera movement filter

Menu-State filter

Subtitle filter

Video quality filter

Human face filter

Aesthetic filter

- Figure 3: We construct our high-quality unlabeled training data from raw gameplay videos via a three-stage hierarchical filtering pipeline.

Motion filtering: To ensure meaningful temporal dynamics, we apply motion filtering using GMFlow [60] to compute average optical flow magnitude per clip. Videos with too little motion (e.g., static screens) or excessive motion (e.g., rapid spinning or scene glitches) are discarded. This bidirectional filtering retains motion-balanced sequences, supporting stable training and improving the model’s ability to learn temporally consistent and controllable video generation.

Camera movement filtering: To remove clips with overly aggressive viewpoint changes, we apply camera motion filtering based on angular changes estimated by the Inverse Dynamics Model (IDM) [4]. Videos with excessive yaw or pitch rotation (often caused by abrupt mouse movements) are discarded. This filtering step promotes stable and coherent viewpoint trajectories, helping the model learn consistent scene geometry and spatial alignment over time.

### 3.2 Labeled Data Creation

Hybrid labeled dataset construction via exploration and simulation. To enable controllable video generation, we construct labeled datasets using two complementary strategies: Exploration Agent trajectories derived from gameplay in the MineRL [38] environment, and Unreal Procedural Simulation, which offers high-fidelity, scriptable environments with precise control annotations.

Exploration agent. We extend the MineRL platform by deploying curriculum-guided VPT agents [4], which are capable of performing long-horizon tasks within the Minecraft world. These agents autonomously explore diverse in-game scenarios, generating a wide range of behavior patterns. We extract per-frame keyboard and mouse actions from these trajectories to construct an action-labeled dataset, sampled at 16Hz. This large-scale dataset provides dense supervisory signals for training control-aware generation models.

Unreal procedural simulation. To complement exploration data with highly structured demonstrations, we build custom environments in Unreal Engine that span various biomes, including urban, desert, and forest settings. Each environment is programmatically designed and instrumented to provide detailed supervision at every frame. Specifically, we collect: (1) Discrete action labels (e.g., movement keys and jump) and continuous gaze vectors (camera pitch/yaw). (2) Ground-truth kinematic information, including the agent’s position, velocity, and orientation. (3) Environmental interaction outcomes, such as the success or failure of block manipulation actions. This procedurally generated data provides consistent, noise-free annotations, enabling the model to learn precise action-response mappings under diverse and controllable conditions.

Curation strategies for high-quality and balanced labeled data. Labeled data plays a pivotal role in ensuring model convergence and enhancing controllability. To ensure that the training data provides strong and reliable supervision signals, we incorporate three key strategies during the construction of labeled Minecraft trajectories using MineRL exploration agents. These strategies are designed to guarantee high visual fidelity, semantic diversity across gameplay scenarios, and balanced distribution across control categories, thereby improving both the robustness and generalization capability of the resulting model.

Camera motion restriction. To ensure viewpoint stability and facilitate the learning of temporally consistent visual representations, we explicitly constrain camera movements during data generation. Specifically, the yaw and pitch angles are restricted to within 15◦ per frame, effectively avoiding

- Table 1: Biome distributions and characteristics in the balanced Matrix-Game-MC dataset for 65frame controllable video generation. The dataset spans 14 Minecraft biomes, each contributing 4–7% of samples, ensuring semantic diversity and covering a range of visual and physical conditions.

Biome Percentage Environmental Features

Forest 4.0% Dense trees, wolves, flowers, mushrooms Taiga 4.5% Spruce trees, foxes, berry bushes, snowfall Swamp 4.6% Mangrove trees, slime blocks, lily pads, vines Ocean 7.2% Coral reefs, kelp forests, drowned ruins, prismarine Mesa 6.7% Red sandstone, hardened clay strata, dead bushes Extreme hills 7.2% Mountain peaks, emerald ore, snowcaps, waterfalls Savanna 6.3% Baobab trees, acacia wood, herds of llamas Plains 6.0% Rolling grasslands, villages, sunflowers, horses Beach 6.5% Sandy shores, turtle nests, sugarcane, shallow waters Jungle 5.9% Giant trees, cocoa plants, ocelots, temple ruins River 5.8% Flowing water, clay deposits, salmon, gravel banks Desert 7.9% Sand dunes, cacti, desert temples, husks Mushroom 6.3% Mycelium terrain, giant mushrooms, mooshrooms Icy 6.8% Icebergs, polar bears, packed ice, strays Random 14% Random spawn for scenarios like Nether/End

sequences with abrupt camera rotations or disorienting viewpoint shifts. Such unstable visual motion, commonly observed in speed, run or chaotic gameplay, can undermine the temporal smoothness of rendered scenes, making it more difficult for the model to learn frame-to-frame coherence and predict stable visual transitions. By enforcing this constraint at the data creation stage, we produce observation sequences that are visually stable, contextually grounded, and easier to model temporally. This design promotes better alignment between scene perception and control conditioning, ultimately enhancing the model’s ability to generate smoother and more coherent videos in response to user inputs.

MineRL engine modification. To ensure visual consistency and eliminate rendering artifacts, we introduce targeted modifications to the MineRL engine during data creation. Specifically, we disable frustum-based chunk loading, a mechanism that causes new terrain blocks to abruptly appear as the camera moves. This change prevents sudden scene pop-ins that would otherwise disrupt the spatial coherence of the visual stream. In addition, we implement real-time monitoring of the agent’s health status and in-game interface state. Recording is automatically terminated when the agent is near death, stuck, or when the pause/menu screen is activated. These safeguards ensure that all captured clips reflect uninterrupted, meaningful gameplay interactions rather than irrelevant or low-quality segments. Together, these modifications help produce high-fidelity video clips that are free of visual artifacts and non-interactive states, providing the model with stable, context-rich supervision signals for learning controllable and temporally coherent world generation.

Scenario diversification. We selectively curate 14 Minecraft scenarios, each tied to a specific biome (e.g., forest, desert, ocean), covering diverse terrains, lighting, and structures. Players perform discrete actions (move, jump, attack) in each setting, sampled in a balanced way. This setup exposes the model to varied spatial and physical contexts, improving its ability to generalize motion dynamics and interaction behavior across different virtual world environments.

Final labeled dataset for controllable world generation. By applying the aforementioned data construction strategies to the MineRL-based exploration agent, we synthesize a high-quality labeled dataset for Minecraft, which forms a substantial portion of our controllable training corpus. To enhance both visual diversity and control fidelity, we further incorporate procedurally generated videos from Unreal Engine. Together, these sources yield a comprehensive labeled dataset containing over 1,026 hours of video clips for 33-frame training.

To ensure balanced coverage across environments, we additionally curate an expanded dataset comprising more than 1,200 hours of video for 65-frame training. Notably, approximately half of this balanced set is sourced from MineRL-based scenarios spanning 14 distinct Minecraft biomes, such as forest, desert, icy, and mushroom, whose distribution is detailed in Table 1. The resulting datasets are motion-stable, action-dense, and structurally diverse, offering strong and balanced supervision signals for training robust, controllable video generation models capable of generalizing across varied virtual environments.

[Figure 22]

[Figure 23]

Interactive control signal

Add noise

[Figure 24]

[Figure 25]

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

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

3DCausal Encoder

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

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

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

###### Matrix-Game DiffusionTransformer

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

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

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

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

…

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

[Figure 178]

[Figure 179]

3DCausal

[Figure 180]

[Figure 181]

[Figure 182]

[Figure 183]

[Figure 184]

[Figure 185]

[Figure 186]

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

[Figure 193]

Decoder

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

[Figure 199]

[Figure 200]

[Figure 201]

[Figure 202]

[Figure 203]

[Figure 204]

[Figure 205]

[Figure 206]

[Figure 207]

[Figure 208]

[Figure 209]

[Figure 210]

[Figure 211]

[Figure 212]

[Figure 213]

[Figure 214]

[Figure 215]

[Figure 216]

[Figure 217]

[Figure 218]

[Figure 219]

[Figure 220]

[Figure 221]

[Figure 222]

[Figure 223]

[Figure 224]

[Figure 225]

[Figure 226]

[Figure 227]

[Figure 228]

[Figure 229]

[Figure 230]

[Figure 231]

[Figure 232]

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

[Figure 249]

[Figure 250]

[Figure 251]

…

[Figure 252]

[Figure 253]

[Figure 254]

[Figure 255]

[Figure 256]

[Figure 257]

[Figure 258]

[Figure 259]

[Figure 260]

[Figure 261]

[Figure 262]

[Figure 263]

[Figure 264]

[Figure 265]

[Figure 266]

[Figure 267]

[Figure 268]

[Figure 269]

Visual Encoder

[Figure 270]

[Figure 271]

[Figure 272]

[Figure 273]

- Figure 4: Overview of the interactive image-to-world generation paradigm. The model is trained in a spatiotemporally compressed latent space obtained through a 3D Causal VAE. Conditioned on a reference image along with Gaussian noise and action signals, it generates latent representations that are decoded into video clips. By grounding generation in the reference image, the model learns to build consistent scene representations that capture geometry, dynamics, and physical interactions, enabling the generation of temporally coherent and spatially structured videos.

## 4 Matrix-Game: An Interactive World Foundation Model

### 4.1 Model Architecture and Design

Exploring spatial intelligence through image-to-world modeling. In light of Cosmos [1] and Genie-2 [41], we construct our world foundation model, i.e., Matrix-Game, using latent diffusion models [45]. Most existing diffusion-based world models, such as SORA [40], HunyuanVideo I2V [29] and Wan [54], rely on text prompts and reference images as prior knowledge to guide the generation process. While these methods can produce high-quality results, the introduction of text often imposes semantic biases, constrains spatial interpretation, and reduces the model’s ability to ground understanding purely in visual and physical cues. As a result, the model may hallucinate unrealistic content or overfit to language priors rather than faithfully modeling the visual world. Inspired by the concept of Spatial Intelligence [57], Matrix-Game explores a different route: instead of using both text and images as conditions, our model learns purely from raw images. It learns to understand the world by building a consistent scene that captures geometry, object movement, and how things interact physically.

As shown in Figure 4, Matrix-Game adopts an image-to-world generation paradigm, using a single reference image as the primary prior for world understanding and video generation. The model is trained in a spatiotemporally compressed latent space constructed by a 3D Causal VAE [28,68], which reduces the spatial and temporal resolution of video sequences by factors of 8 and 4, respectively. The reference image, processed by either a visual encoder or a multi-modal backbone, serves as the central conditioning input. Conditioned on Gaussian noise and optional user actions, a Diffusion Transformer (DiT) generates latent representations, which are then decoded into coherent video sequences via the 3D VAE decoder.

Unlike prior methods that treat the image as just an auxiliary input alongside text prompts, MatrixGame is trained to perceive, interpret, and model the world directly from visual content alone. Through large-scale training on diverse game-world environments, the model learns to recognize the spatial layout of scenes, how objects move, and how they physically interact. This allows it to build a consistent and structured understanding of the world from a single image, which forms the basis for generating realistic and controllable videos.

By combining visual understanding with user control, Matrix-Game transforms video generation into an interactive tool for exploration and creation, empowering users to see, modify, and build coherent virtual worlds starting from a single image.

Token replace

[Figure 274]

[Figure 275]

[Figure 276]

[Figure 277]

[Figure 278]

[Figure 279]

Motion Frames Noisy Input Reference Image

Auto-regressive Generation

[Figure 280]

[Figure 281]

[Figure 282]

[Figure 283]

[Figure 284]

[Figure 285]

[Figure 286]

[Figure 287]

[Figure 288]

[Figure 289]

[Figure 290]

[Figure 291]

3DCausal Encoder

Visual Encoder Channel concat with masking

3D Causal Encoder

[Figure 292]

[Figure 293]

[Figure 294]

[Figure 295]

[Figure 296]

[Figure 297]

[Figure 298]

[Figure 299]

[Figure 300]

[Figure 301]

[Figure 302]

[Figure 303]

[Figure 304]

[Figure 305]

[Figure 306]

…

…

C

[Figure 307]

Motion Frames

Control Signal

[Figure 308]

[Figure 309]

[Figure 310]

[Figure 311]

###### Token Refiner

Patch Embedding

[Figure 312]

[Figure 313]

[Figure 314]

[Figure 315]

[Figure 316]

[Figure 317]

[Figure 318]

[Figure 319]

[Figure 320]

[Figure 321]

[Figure 322]

[Figure 323]

[Figure 324]

[Figure 325]

[Figure 326]

[Figure 327]

[Figure 328]

[Figure 329]

…

[Figure 330]

Control Signals

[Figure 331]

…

###### Matrix-Game DiffusionTransformer

[Figure 332]

[Figure 333]

[Figure 334]

|[Figure 335]<br><br>[Figure 336]<br><br>[Figure 337]<br><br>Double-stream DiT Blocks Single-stream DiT Blocks<br><br>[Figure 338]<br><br>[Figure 339]<br><br>[Figure 340]<br><br>[Figure 341]<br><br>|
|---|

[Figure 342]

[Figure 343]

[Figure 344]

[Figure 345]

[Figure 346]

…

3DCausal

[Figure 347]

Decoder

[Figure 348]

[Figure 349]

[Figure 350]

[Figure 351]

Noisy Latent

[Figure 352]

[Figure 353]

[Figure 354]

[Figure 355]

[Figure 356]

…

[Figure 357]

MLP

…

[Figure 358]

[Figure 359]

[Figure 360]

Visual Encoder

[Figure 361]

[Figure 362]

[Figure 363]

[Figure 364]

[Figure 365]

[Figure 366]

[Figure 367]

[Figure 368]

…

[Figure 369]

[Figure 370]

[Figure 371]

Sinusoidal Encoding

[Figure 372]

[Figure 373]

[Figure 374]

[Figure 375]

[Figure 376]

[Figure 377]

Timestep

Output

Reference Image

(a) Autoregressive generation in Matrix-Game

(b) The architecture of Matrix-Game

- Figure 5: (a) Autoregressive generation in Matrix-Game and (b) The architecture of Matrix-Game. To enable long-duration video generation, Matrix-Game adopts an autoregressive strategy: the last few frames of each generated clip are used as motion conditions for generating the next clip. Specifically, the latent of these motion frames are concatenated with the noisy latent along the channel dimension, and a binary mask is also concatenated to indicate which frames contain valid motion information. This design enhances local temporal consistency across video segments, allowing the model to maintain coherent dynamics over extended time horizons. Moreover, we adopt the token replacement trick in HunyuanVideo I2V [29] to enable stable image-to-video generation.

Autoregressive generation with diffusion transformer. To enable high-quality generation, following Flux [30] and HunyuanVideo [29], we adopt a multi-modal diffusion transformer (MMDiT) for image-to-world modeling. Like most current video generation methods [29,40,54], the image-toworld model, as illustrated in Figure 4, generates fixed-length video clips, which limits its applicability in real-world scenarios that demand long-term or continuous world modeling. To address this limitation, and drawing inspiration from recent advances in long-duration video generation [51,73], we adopt an autoregressive strategy: at each step, the model takes the previously generated video clip as motion context to produce the next. As shown in Figure 5(a), we use the last k = 5 frames from each generated segment as motion conditions for generating the subsequent clip. Such a design enables the model to progressively extend generation over time while preserving temporal coherence across segments.

To this end, as illustrated in Figure 5(b), we concatenate the latent representations of the motion frames with the noisy latent along the channel dimension to form the input for the next generation step. A binary mask is concatenated to indicate which frames contain valid motion information. The combined latent tensor is then processed through a patch embedding layer and further concatenated with the image tokens along the token dimension. Finally, conditioned on user control signals as an additional guidance input, the multi-modal diffusion transformer generates a new video clip.

However, a key challenge in autoregressive generation is temporal error accumulation: artifacts in the last few generated frames can propagate and amplify in subsequent segments. To improve the robustness of the autoregressive process, inspired by the Open-Sora plan [33], we introduce Gaussian noise to the motion frames and reference images with a probability of 0.2 during training. In addition, we apply classifier-free guidance (CFG) to motion frames during training: the latents of motion frames are replaced with unconditioned signals (i.e., zero latents) with a probability of 0.25. This CFG strategy encourages the model to rely more effectively on motion context, leading to more stable and reliable autoregressive video generation.

Injecting actions for controllable video generation. Inspired by Genie 2 [41], we employ framelevel control signals to guide video generation. As shown in Figures 6(a-b), we integrate a control module into our multi-modal diffusion transformer to enable action-controllable generation, following the action control design introduced in GameFactory [67]. The detailed architecture of this action control module is shown in Figure 6(c).

[Figure 378]

[Figure 379]

[Figure 380]

[Figure 381]

[Figure 382]

[Figure 383]

[Figure 384]

[Figure 385]

[Figure 386]

[Figure 387]

[Figure 388]

[Figure 389]

[Figure 390]

[Figure 391]

[Figure 392]

[Figure 393]

[Figure 394]

[Figure 395]

[Figure 396]

[Figure 397]

[Figure 398]

[Figure 399]

[Figure 400]

[Figure 401]

…

…

…

…

[Figure 402]

[Figure 403]

[Figure 404]

[Figure 405]

[Figure 406]

[Figure 407]

Norm Norm

[Figure 408]

Norm

[Figure 409]

[Figure 410]

[Figure 411]

[Figure 412]

[Figure 413]

[Figure 414]

[Figure 415]

Scale & Shift Scale & Shift

C

[Figure 416]

Mouse action

[Figure 417]

[Figure 418]

[Figure 419]

[Figure 420]

[Figure 421]

Scale & Shift

QK-Norm QK-Norm

MLP

[Figure 422]

[Figure 423]

[Figure 424]

[Figure 425]

Attention (3D RoPE)

Attention (3D RoPE)

Mouse-driven Self-attention

[Figure 426]

[Figure 427]

[Figure 428]

[Figure 429]

[Figure 430]

Control Module

QK-Norm

[Figure 431]

[Figure 432]

[Figure 433]

[Figure 434]

[Figure 435]

[Figure 436]

Attention (3D RoPE)

Gate Gate

[Figure 437]

+

[Figure 438]

[Figure 439]

MLP

[Figure 440]

[Figure 441]

[Figure 442]

[Figure 443]

+ +

[Figure 444]

[Figure 445]

Control Norm Module

[Figure 446]

[Figure 447]

[Figure 448]

[Figure 449]

V K Q

Norm

Keyboard action

[Figure 450]

[Figure 451]

Keyboard-driven Cross-attention

C

[Figure 452]

[Figure 453]

[Figure 454]

[Figure 455]

Scale & Shift Scale & Shift

[Figure 456]

[Figure 457]

C

[Figure 458]

[Figure 459]

[Figure 460]

[Figure 461]

MLP MLP

[Figure 462]

[Figure 463]

[Figure 464]

Gate

+

[Figure 465]

[Figure 466]

[Figure 467]

[Figure 468]

Gate

Gate

[Figure 469]

[Figure 470]

[Figure 471]

[Figure 472]

[Figure 473]

[Figure 474]

+ +

+

(a) Double-stream DiT Block (b) Single-stream DiT Block

(c) Control Module

Figure 6: The details of diffusion transformer blocks in Matrix-Game.

Specifically, we use discrete encodings to represent keyboard actions, including “up,” “down,” “left,” “right,” “jump,” and “attack”, and continuous scalar values to represent mouse movements, defined as changes in pitch angle. To align these action signals with the compressed latent tokens produced by the 3D Causal VAE, we adopt the group operation trick [67], which accounts for the temporal compression ratio and enables effective token-level conditioning. The continuous mouse action is concatenated with the input latent and processed through an MLP followed by temporal self-attention, while the discrete keyboard actions are integrated via cross-attention to guide the diffusion process. We also apply classifier-free guidance to action signals during training by replacing them with unconditioned signals with a probability of 0.1. This helps the model learn to use action signals more effectively when they are provided, resulting in better control and interaction in the generated videos.

### 4.2 Model Training

To improve training stability and enable faster inference, Matrix-Game leverages the flow matching paradigm [11], which outperforms traditional denoising diffusion probabilistic models (DDPM) [23] in both convergence and sampling efficiency. Based on this paradigm, we adopt the rectified flow loss [36] for model training. To support complex tasks such as modeling world knowledge, capturing physical dynamics, and enabling action-controllable generation, we organize the training process into two progressive stages, each optimized for distinct learning objectives.

- Stage 1: Unlabeled training for game world understanding. To accelerate convergence, the model is initialized with pretrained weights from the HunyuanVideo image-to-video model [29]. To shift from text-driven generation to image-conditioned world modeling, we replace the original text branch in the multi-modal diffusion transformer with an image branch, as illustrated in Figure 5(b). The action control module is excluded in this stage to focus on visual world understanding.

The primary goal is to pretrain the model on large-scale game environments, enabling it to build structured understanding of virtual worlds, including spatial layouts, object dynamics, and intuitive physical rules. To this end, we use 2,700-hour unlabeled Minecraft videos at 720p resolution as a rich source of visual and physical cues. We train the model using a diverse mixture of frame counts (17, 33, and 65) and aspect ratios (16:9, 4:3, and 21:9) to enhance robustness under varied temporal and spatial settings.

Following the initial large-scale pretraining, we further refine the model’s visual and physical understanding by curating a subset of 870-hour high-quality video clips from the same dataset. These clips are selected based on stable camera motion, clean user interfaces, and overall visual clarity. This targeted refinement improves Matrix-Game’s ability to model coherent spatial structures, capture fine-grained physical interactions, and generate videos with higher perceptual quality and temporal consistency.

- Stage 2: Action-labeled training for interactive world generation. In the second stage, we integrate the action control module into a multi-modal diffusion transformer to enable action-controllable video generation. The final model, Matrix-Game, contains 17 billion parameters and is trained on 1,200 hours of action-labeled 720p 33-frame video clips collected from both Minecraft and Unreal Engine environments. To ensure training stability and efficiency during controllable generation, we adopt a fixed 720p resolution and a 33-frame setting in the early training phase.

To mitigate class imbalance in world scenarios [70,71], we further refine the training data in the second sub-stage. Specifically, we curate a more balanced dataset by organizing samples across 8 distinct Minecraft biomes: beach, desert, forest, hills, icy, mushroom, plains, and river. Combined with procedurally generated data from Unreal Engine, this results in a high-quality, balanced training set comprising approximately 1,200 hours of 720p 65-frame video clips.

We continue training under the 65-frame setting to strengthen the model’s ability to capture long-range temporal dependencies, which are essential for maintaining coherent interaction across extended sequences. Through this integration of balanced, action-rich data and strong visual priors, MatrixGame learns to precisely interpret user inputs and generalize across diverse interactive environments. This tight coupling of visual understanding and user control advances video generation into an interactive paradigm for world exploration and creation, empowering users to perceive, modify, and construct coherent virtual environments from a single reference image.

## 5 GameWorld Score: A Unified Benchmark for Minecraft World Models

With the rise of world models, an increasing number of studies have focused on the Minecraft world generation, aiming to leverage video generation models to produce videos that not only align with user action inputs, but also adhere to the physical rules inherent in the game. However, existing research lacks a unified evaluation benchmark to consistently measure and compare model performance in the setting with action inputs. Current benchmarks, such as VBench [25], primarily evaluate visual quality and the alignment between generated videos and textual prompts. WorldScore [10] goes a step further by assessing the ability of models to generate 3D/4D worlds conditioned on textual inputs. However, these benchmarks remain limited in evaluating image-to-world generation, and are inadequate for scenarios involving fine-grained control signals, such as frame-by-frame action conditions commonly found in real gameplay settings.

Minecraft world generation involves a combination of discrete control signals (e.g., move forward, move backward, move left, move right, jump, attack, etc.) and continuous control inputs (e.g., camera rotation along the X and Y axes). Therefore, a comprehensive benchmark needs to accommodate these types of controls and offer corresponding evaluations. Furthermore, Minecraft follows fundamental physical laws, such as object consistency and scenario consistency. A capable world model should generate outputs that are consistent with these physical intuitions.

To better measure and compare Minecraft world models, we develop GameWorld Score, a unified benchmark that evaluates not only the perceptual quality of generated videos, but also their controllability and physical plausibility. Specifically, we decompose the evaluation of world model performance into eight dimensions, each targeting a distinct aspect of video generation. At the top level, GameWorld evaluates models across four key pillars:

Visual quality: Assesses each individual frame’s visual fidelity in alignment with the Human Visual System (HVS), focusing on clarity, coherence, and realism of still images.

Temporal quality: Measures how well the model maintains consistency and smoothness over time, capturing dynamics like motion continuity and temporal coherence.

Action controllability: Evaluates whether the generated video faithfully follows the user-provided control inputs, such as movement commands and camera adjustments.

[Figure 475]

[Figure 476]

[Figure 477]

[Figure 478]

[Figure 479]

[Figure 480]

[Figure 481]

- Figure 7: GameWorld Score provides a unified benchmark for assessing the quality and realism of generated Minecraft worlds.

Physical rule understanding: Evaluates whether the video adheres to fundamental physical principles, with a particular focus on maintaining object consistency and scenario consistency across space and time, reflecting the model’s ability to simulate physically coherent environments.

These four pillars are further divided into fine-grained dimensions, as illustrated in Figure 7, enabling a comprehensive and structured assessment of generative models within interactive and physics-driven environments like Minecraft.

### 5.1 Visual Quality

Frame-wise quality measures the merit of individual frames, ignoring their interplay across time. We analyse every frame from two complementary perspectives:

Aesthetic quality. We evaluate the perceived visual appeal of individual frames using the LAION aesthetic predictor [31], a learned model trained on large-scale human aesthetic preferences. This score reflects a combination of factors, including image composition, color harmony, lighting balance, photorealism, and stylistic coherence. A higher aesthetic score indicates stronger alignment with human judgments of visual attractiveness, providing a complementary perspective beyond pixel-level or structural metrics.

Image quality. To assess the perceptual fidelity of each frame, we evaluate low-level visual artifacts such as over-exposure, noise, compression distortions, and blurriness using the MUSIQ predictor [26]. MUSIQ is a no-reference image quality assessment model trained on the SPAQ dataset [13], which reflects diverse real-world imaging conditions. This metric provides a quantitative measure of how clean, sharp, and artifact-free the generated frames appear, serving as a robust proxy for humanperceived visual quality.

### 5.2 Temporal Quality

Temporal Quality evaluates how well the generated video maintains consistency and realism across consecutive frames. This aspect is crucial for ensuring coherent motion, avoiding flickering artifacts, and preserving object integrity over time. To capture different facets of temporal stability, we propose two complementary dimensions:

Temporal consistency. To evaluate how well the background and scene remain stable over time, we compute the pairwise similarity of CLIP [44] features extracted from each frame in the video sequence. CLIP embeddings capture high-level semantic and visual information, making them suitable for assessing whether consecutive frames depict a temporally coherent scene. Specifically, we calculate the average cosine similarity between adjacent frames to quantify consistency. Higher similarity indicates that the model preserves static elements such as background layout across time, thus avoiding common artifacts like flickering, texture drift, or abrupt visual changes.

Motion smoothness. While temporal consistency ensures stable appearance across frames, it does not account for the quality of motion itself. Abrupt or jittery transitions may still occur even

when frame content appears coherent. To address this, we assess motion smoothness by evaluating whether the movement of objects and camera follows physically plausible and temporally continuous trajectories. Following [32], we leverage the motion priors learned by a video frame interpolation network to detect unnatural dynamics. Specifically, we feed the generated video into a pretrained interpolation model [32] and measure the reconstruction error between actual frames and those interpolated from adjacent frames. High interpolation accuracy implies that motion flows smoothly between frames, while large discrepancies indicate irregularities such as jitter, stuttering, or framelevel discontinuities. This method provides a proxy for judging motion realism without requiring dense annotations.

### 5.3 Action Controllability

This module aims to assess how well the generated videos understand and follow the input action conditions. Ideally, the visual content of generated videos should respond frame-by-frame to the given player-like control signals, mirroring the behavior of an interactive game environment. In our setup, we categorize the input signals into two types: keyboard conditions and mouse conditions. Keyboard conditions govern directional movement (forward, back, left, right) as well as jump and attack actions. Mouse conditions control the camera orientation in eight directions (up, down, left, right, upper-left, upper-right, lower-left, and lower-right), represented by rotational angle changes.

Following MineWorld [18], we adopt the Inverse Dynamics Model (IDM) to assess controllability by inferring the underlying action conditions from a given video sequence. Controllability is measured by comparing the inferred actions with ground truth inputs, evaluating how accurately the generated video reflects intended control signals. Please note that IDM, trained on 1,962 hours of Minecraft gameplay, achieves 90.6% accuracy on keyboard prediction and an R2 score of 0.97 for mouse movement regression, making it a reliable proxy for extracting action labels from video.

Keyboard control accuracy. We evaluate the controllability of keyboard inputs by computing precision across four grouped action categories: (forward, back, empty), (left, right, empty), (attack, empty), and (jump, empty). Each group is treated as a multi-class classification problem with mutually exclusive actions. The final keyboard condition accuracy is reported as the average precision across these four groups. In addition to this aggregated score, we also report the per-class precision for each individual action (e.g., forward, left, jump, etc.), which provides a more fine-grained analysis of how well the model responds to different types of control inputs. This allows us to analyze how well the model follows specific control commands.

Mouse control accuracy. Mouse inputs affect the camera’s rotational movement, which is modeled independently from keyboard actions. For each axis (x and y), directional movement is detected when the absolute rotational change exceeds a pre-defined threshold. This results in nine categories: up, down, left, right, upper-left, upper-right, lower-left, lower-right, and empty. A prediction is considered correct if the motion direction in the generated video matches the labeled condition. The final accuracy is reported as the precision over all positive predictions.

### 5.4 Physical Rule Understanding

To assess the model’s understanding of the physical world rule, we evaluate its ability to preserve Object Consistency and Scenario Consistency across frames.

Object Consistency. A physically grounded model should preserve object geometry over time despite texture changes. Following WorldScore [10], we use DROID-SLAM [50] to estimate depth and camera pose, and compute the reprojection error between co-visible pixels across frames. Since DROID-SLAM is robust to appearance changes, this metric isolates geometric consistency at the object level. Lower error indicates better preservation of object structure across time.

Scenario Consistency. We propose a Scenario Consistency metric to evaluate how well a model preserves the overall scene over time. It uses 8 types of symmetric camera motion—e.g., up-thendown, left-then-right, and diagonal pairs—where the camera moves in one direction and then reverses along the same path. Corresponding frames in each direction should ideally match. The motion is large enough to move the scene out of view, forcing the model to recover it during the return. Consistency is measured by the mean-squared error (MSE) between paired frames, allowing up to 4-pixel shifts to handle minor misalignments. Lower error indicates better scenario consistency.

- Table 2: Comparisons of Minecraft world generation in terms of the GameWorld Score benchmark. Keyboard Acc. and Mouse Acc. represent the accuracy of control signal prediction for keyboard actions and mouse-driven camera movements, respectively. Motion smooth. evaluates the smoothness of frame-wise motion transitions, while Temporal Cons. measures short-range temporal consistency. Obj. Cons. reflects the 3D consistency of objects across space and time, while Scenario Cons. indicates the the consistency of scenarios across time.

Visual Quality Temporal Quality Action Controllability Physical Understanding Image Quality ↑ Aesthetic ↑ Temporal Cons. ↑ Motion smooth. ↑ Keyboard Acc. ↑ Mouse Acc. ↑ Obj. Cons. ↑ Scenario Cons. ↑

Model

Oasis [9] 0.65 0.48 0.94 0.98 0.77 0.56 0.56 0.86 MineWorld [18] 0.69 0.47 0.95 0.98 0.86 0.64 0.51 0.92 Ours 0.72 0.49 0.97 0.98 0.95 0.95 0.76 0.93

## 6 Experiments

Experimental objectives. Our experiments are designed to comprehensively evaluate the proposed model across multiple dimensions. Specifically, we aim to answer the following four questions: (1) GameWorld Score Benchmark: Does our model outperform existing state-of-the-art open-source Minecraft models across key dimensions such as visual quality, temporal quality, action controllability, and physical rule understanding? (2) Action Controllability: How well does our model respond to various user commands, particularly keyboard actions and mouse movements? (3) Scene Generalization: How does the model perform across diverse Minecraft scenarios (e.g., forest, desert, icy, mushroom)? (4) Autoregressive Generation: Can our model maintain coherent and controllable behavior during long-horizon autoregressive video generation?

Implementation details. The experiments were conducted with a per-GPU batch size of 1. We employed bf16 mixed precision and the Fully Sharded Data Parallel (FSDP) strategy for efficient large-scale training. The learning rate was set to 5 × 10−5, with 16 training FPS and 5 motion frames. During inference, we applied Classifier-Free Guidance (CFG) to the reference image, motion frames, and action signals. The CFG scale was set to 6, with 50 sampling steps using Flow Matching. The flow matching shift parameter was set to 15.

Compared methods. To establish a solid comparison, we include two of the most representative open-source world models as baselines: OASIS [9] and MineWorld [18]. Both are recent works released with code and models available, and have shown competitive results in Minecraft world generation. These models provide a reasonable benchmark for evaluating visual quality, temporal dynamics, and controllability, allowing us to position our method in relation to existing publicly available systems.

Evaluation metrics. We evaluate Minecraft world generation performance using the proposed GameWorld Score benchmark (cf. Section 5). Moreover, to complement standard quantitative metric, which often fail to capture subtle differences in perceptual quality, we further conduct human evaluation through manual scoring on all baseline outputs. This evaluation covers four key aspects: Overall Quality, Controllability, Visual Quality, and Temporal Consistency. It is performed in a double-blind setting with two independent annotator groups, where neither group is aware of the method identities, ensuring fairness and minimizing potential bias.

### 6.1 Model Performance

GameWorld Score benchmark. As shown in Table 2 and Figure 2, Matrix-Game achieves strong and well-rounded performance across all dimensions of the GameWorld Score benchmark. Compared to leading open-source baselines such as OASIS [9] and MineWorld [18], our model excels especially in two key aspects: controllability and physical consistency. It responds more accurately to control inputs, including both keyboard and mouse signals, which is critical for interactive world generation. Additionally, it better preserves the object and scenarios of the game world over time, demonstrating a stronger understanding of physical layout and geometry. Beyond these, our Matrix-Game also maintains competitive performance in other key aspects. It produces sharper and more aesthetically pleasing visuals, as reflected in both the image quality and aesthetic scores. Furthermore, it achieves high temporal consistency and motion smoothness, ensuring seamless transitions between frames without flickering or abrupt changes. Altogether, these results highlight the model’s ability to deliver high-quality, physically grounded, and user-controllable video outputs suitable for complex interactive world generation.

Oasis MineWorld Ours

| |0.4% 0.6% 0.4% 0.4%<br><br>3.3% 5.7%<br><br>1.4%<br><br>10.0%<br><br>96.3%<br><br>93.8%<br><br>98.2%<br><br>89.6%| | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

1.0

0.8

Winrate

0.6

0.4

0.2

0.0

OverallQuality Controllability VisualQuality TemporalConsistency

- Figure 8: Human evaluation results (double-blind setting) across four predefined dimensions: Overall Quality, Controllability, Visual Quality, and Temporal Consistency. The win rate reflects the proportion of scenario-metric pairs in which each method is rated as the best by annotators.

- Table 3: Comparison of control accuracy. The evaluation covers two types of control signals: keyboard actions and camera movements. Camera movement is categorized into eight directions based on pitch and yaw angle changes: ↑, ↓, ←, →, and their diagonals (↖, ↗, ↙, ↘). Accuracy is computed using an Inverse Dynamics Model (IDM) by comparing predicted actions from generated videos with ground truth labels. Higher accuracy values represent better controllability.

Keyboard Action Mouse Movement Action forward backward left right jump attack camera↑ camera↓ camera← camera→ camera↖ camera↗ camera↙ camera↘

Model

Oasis [9] 0.85 0.78 0.80 0.79 0.77 0.89 0.66 0.55 0.33 0.35 0.56 0.53 0.45 0.51 MineWorld [18] 0.86 0.80 0.87 0.88 0.82 0.87 0.46 0.45 0.53 0.54 0.66 0.77 0.87 0.96 Ours 0.99 0.91 0.92 0.96 0.88 0.95 0.91 0.98 0.89 0.90 0.92 0.97 0.98 0.98

Human studies. To complement objective metrics and mitigate potential biases, we conducted human evaluation through two independent double-blind studies. Each study was carried out by a separate group of annotators, who were unaware of the method identities and each other’s assessments. Annotators compared video outputs across four key dimensions: Overall Quality, Controllability, Visual Quality, and Temporal Consistency. As shown in Figure 8, our method is overwhelmingly favored by human evaluators across all assessed dimensions. It achieves a 96.30% win rate in Overall Quality, indicating that users consistently preferred our results when evaluating the overall impression of the generated videos, including realism, coherence, and completeness. In terms of Controllability, our method scores 93.76%, reflecting its strong ability to accurately translate keyboard and mouse inputs into desired in-game behaviors. This highlights its superiority in generating interactive content with precise user control. Our approach also achieves 98.23% in Visual Quality, showcasing clear advantages in perceptual fidelity, texture clarity, and aesthetic appeal of individual frames. Finally, with 89.56% in Temporal Consistency, our method demonstrates robust temporal coherence, ensuring that motion remains stable and physically plausible across frames. Importantly, these human evaluation results, based solely on subjective preference, align with our proposed GameWorld Score benchmark, reinforcing its reliability as an effective proxy for human judgment.

### 6.2 Action Controllability

To rigorously evaluate the action control capabilities, we compare Matrix-Game with OASIS [9] and MineWorld [18] across two control modalities: keyboard actions (e.g., forward, backward, left, right, jump, and attack) and mouse-based camera movements (e.g., yaw and pitch angle shifts in 8 directions). As shown in Table 3, Matrix-Game consistently achieves the highest control accuracy across all action types. Regarding keyboard control, our model attains over 88% accuracy on all actions, with particularly strong performance in fine-grained directions such as “forward” (99%), “right” (96%), and “jump” (95%). For mouse-based camera movement, typically more challenging due to subtle pitch and yaw transitions, our model still maintains high accuracy, reaching over 89% on all directions. These results highlight our model’s fine-grained controllability and strong alignment between action signals and generated behavior, making it well-suited for interactive world generation and agent conditioning in dynamic environments. For more qualitative results and video demonstrations, please refer to Figures 12-14 and our project website at https:// matrix-game-homepage.github.io.

Scenario Consistency

Scenario Consistency

Scenario Consistency

Scenario Consistency

| |Object|
|---|---|
|0|Image<br><br>20 40 60 80 100|

| |Object|
|---|---|
|0|Image<br><br>20 40 60 80 100|

| |Object|
|---|---|
|0|Image<br><br>20 40 60 80 100|

| |Object|
|---|---|
|0|Image<br><br>20 40 60 80 100|

Mouse Accuracy

Consistency

Mouse Accuracy

Consistency

Mouse Accuracy

Consistency

Mouse Accuracy

Consistency

Keyboard Accuracy

Aesthetic

Keyboard Accuracy

Aesthetic

Keyboard Accuracy

Aesthetic

Keyboard Accuracy

Aesthetic

Motion Smoothness

Quality

Motion Smoothness

Quality

Motion Smoothness

Quality

Motion Smoothness

Quality

Temporal Consistency

Temporal Consistency

Temporal Consistency

Temporal Consistency

Oasis MineWorld Ours

Oasis MineWorld Ours

Oasis MineWorld Ours

Oasis MineWorld Ours

(a) Desert

(b) Beach

(c) Forest

##### (d) Hills

Scenario Consistency

Scenario Consistency

Scenario Consistency

Scenario Consistency

| |Object|
|---|---|
|0|Image<br><br>20 40 60 80 100|

| |Object|
|---|---|
|0|Image<br><br>20 40 60 80 100|

| |Object|
|---|---|
|0|Image<br><br>20 40 60 80 100|

| |Object|
|---|---|
|0|Image<br><br>20 40 60 80 100|

Mouse Accuracy

Consistency

Mouse Accuracy

Consistency

Mouse Accuracy

Consistency

Mouse Accuracy

Consistency

Keyboard Accuracy

Aesthetic

Keyboard Accuracy

Aesthetic

Keyboard Accuracy

Aesthetic

Keyboard Accuracy

Aesthetic

Motion Smoothness

Quality

Motion Smoothness

Quality

Motion Smoothness

Quality

Motion Smoothness

Quality

Temporal Consistency

Temporal Consistency

Temporal Consistency

Temporal Consistency

Oasis MineWorld Ours

Oasis MineWorld Ours

Oasis MineWorld Ours

Oasis MineWorld Ours

(e) Icy

(f) Plain

(g) River

(h) Mushroom

- Figure 9: GameWorld Score across eight scenarios. Each radar chart shows performance over 8 axes: Image Quality, Aesthetic (scaled ×2 for clarity), Temporal Consistency, Motion Smoothness, Keyboard/Mouse Accuracy, and Object/Scenario Consistency. Our method consistently shows superior controllability and physical consistency while preserving high visual and temporal quality.

### 6.3 Scenario Generalization

To assess the generalization capability of our model across diverse game environments, we evaluate it on eight distinct Minecraft scenarios: beach, desert, forest, hills, icy, mushroom, plains, and river. These scenarios cover a wide range of terrain types, object distributions, and interaction dynamics, providing a comprehensive testbed for evaluating world generation performance. As shown in Figure 9, our model consistently outperforms existing open-source baselines, OASIS [9] and MineWorld [18], across all eight scenarios.

Notably, Matrix-Game achieves substantial improvements in action controllability, measured via keyboard and mouse accuracy, and in physical consistency, which assesses the model’s ability to maintain object and scenario consistency over time. These gains underscore the model’s robustness in responding accurately to diverse and fine-grained control signals while preserving geometric coherence across a wide range of world scenarios. Beyond these core capabilities, our model also demonstrates strong performance in visual quality, achieving higher scores in both image fidelity and aesthetic appeal. This ensures that generated frames are not only semantically consistent but also visually compelling. Furthermore, the model maintains superior temporal smoothness, producing videos that transition fluidly between frames with minimal flicker or abrupt motion artifacts. Together, these results highlight the comprehensive generalization ability of Matrix-Game, making it well-suited for complex and dynamic tasks in interactive world generation.

For more qualitative demonstrations across various game scenarios, please refer to Figures 1, 15 and visit our project website at https://matrix-game-homepage.github.io.

### 6.4 Autoregressive Generation for Long Videos

To support long-form video generation, we design our model to operate in an autoregressive multisegment setting, where the video is generated in consecutive segments conditioned on previously synthesized frames. As shown in Figure 10, our approach maintains strong local temporal consistency between segments, effectively bridging the boundaries between consecutive chunks. This results in smooth and visually coherent long-range video outputs without geometry misalignment or abrupt motion shifts, thus can sustain extended interactions and dynamic control behaviors across time.

Please refer to the video demos of long-form video generation at our project website: https: //matrix-game-homepage.github.io.

[Figure 482]

[Figure 483]

[Figure 484]

- (a) Keyboard actions: forward → left → forward

[Figure 485]

[Figure 486]

[Figure 487]

- (b) Keyboard actions: forward → right → forward

[Figure 488]

[Figure 489]

[Figure 490]

(c) Mouse movement actions: (camera →) → (camera ↑) → (camera ↗)

- Figure 10: Auto-regressive generation results of Matrix-Game across three action-conditioned segments. Each segment is generated independently based on the preceding motion context and current action signal. Despite the segment-wise generation, Matrix-Game maintains strong temporal consistency between segments and faithfully follows user-specified control across long sequences.

### 6.5 Discussions on Failure Cases

Edge case generalization. While Matrix-Game demonstrates strong generalization across a wide range of Minecraft scenarios, we do observe certain failure cases in visually complex or underrepresented scenarios. As shown in Figure 11(a), the model may occasionally struggle with precise controllability or spatial consistency in rare biomes or edge cases, typically due to insufficient data coverage. We believe these are natural challenges arising from the open-ended and procedurally generated nature of Minecraft. Although our dataset construction emphasizes diversity and balance, it is inherently difficult to cover all possible world variations. To improve generalization in such cases, we plan to augment our training datasets by scaling up the unlabeled gameplay video clips and creating more controllable data across diverse Minecraft scenes. We also envision adopting a continual training strategy to gradually adapt the model to newly encountered scenarios over time.

[Figure 491]

(a) Edge case generalization

[Figure 492]

(b) Physics understanding

- Figure 11: Failure cases of Matrix-Game. (a) Edge case: the model may fail to maintain temporal consistency in underrepresented or unfamiliar scenarios. (b) Physics understanding: the agent walks through leaves, indicating that there is room to improve the modeling of physical interaction.

Physics understanding. Although our model performs well in terms of object and consistency consistency, there is still room to further enhance its understanding of physical dynamics, particularly in interactions such as object collisions or terrain traversal. For example, as shown in Figure 11(b), we observe cases where the generated agent walks through leaves. These issues are partly due to the scarcity of high-fidelity, physics-supervised data in the current dataset. We believe these aspects can be further improved with richer, physics-aware training data and more explicit modeling of environment constraints. In future work, we plan to curate more physics-aware scenarios, particularly those involving object collisions, support, and material interactions, and extend the GameWorld Score benchmark to include targeted assessments of physical rule understanding. We see this as an important next step toward building models that are not only controllable and coherent, but also physically grounded.

## 7 Conclusions and Future Work

In this work, we introduce Matrix-Game, a novel world foundation model tailored for interactive video generation in open-ended game environments. Alongside the model, we construct the MatrixGame-MC dataset, a large-scale and richly annotated corpus designed to support action-controllable generation in Minecraft-style environments. To facilitate standardized evaluation in this emerging field, we also develop GameWorld Score, a comprehensive benchmark that captures key aspects of perceptual quality, temporal coherence, controllability, and physical consistency. We will release both the model weights and benchmark toolkit to the community, with the goal of advancing future research in interactive world generation.

Future Work. While our method demonstrates strong results across a wide range of scenarios, it is not without limitations. As discussed in Section 6.5, our model occasionally faces challenges in generalization and physical rule understanding, particularly in visually rare or structurally complex environments, largely due to limited training coverage. Addressing these limitations will require expanded data collection, targeted scenario enrichment, and continual training. Beyond this, we highlight three promising directions for future work below.

Long-term temporal consistency. Maintaining coherence over extended video sequences remains a fundamental challenge for interactive world generation. While some recent work [17,59] has attempted to address this issue, current solutions remain for further exploration. We plan to enhance our model architecture by incorporating longer motion contexts or designing memory-based mechanisms to improve consistency across long-range temporal consistency.

Action space enrichment. Our current approach supports six types of keyboard actions and a mouse control space with a limited range of directional values. However, real-world game environments, especially in Minecraft, feature a richer and more nuanced mouse spectrum. We aim to expand our keyboard action space and enable mouse control with a broader and more continuous value range, thereby improving the precision and expressiveness of user interactions.

Beyond Minecraft. While Minecraft provides a well-suited platform for scalable data construction and grounded interaction, it remains a simplified sandbox world. To push the boundaries of controllable generation, we plan to extend our framework to more complex game environments such as Black Myth: Wukong, racing simulators, and multi-agent combat games like CS:GO. These platforms offer richer visual dynamics, complex action semantics, and multi-agent interactions, posing exciting new challenges and opportunities for next-generation world models.

We hope that our contributions, spanning model design, dataset construction, evaluation protocols, and open-source resources, can provide a solid foundation and inspire further progress toward general-purpose, controllable, and physically grounded world generation.

## References

- [1] Niket Agarwal, Arslan Ali, Maciej Bala, Yogesh Balaji, Erik Barker, Tiffany Cai, Prithvijit Chattopadhyay, Yongxin Chen, Yin Cui, Yifan Ding, et al. Cosmos: world foundation model platform for physical ai. arXiv preprint arXiv:2501.03575, 2025.
- [2] Eloi Alonso, Adam Jelley, Vincent Micheli, Anssi Kanervisto, Amos J Storkey, Tim Pearce, and François Fleuret. Diffusion for world modeling: Visual details matter in atari. In Advances in Neural Information Processing Systems, volume 37, pages 58757–58791, 2024.
- [3] Youngmin Baek, Bado Lee, Dongyoon Han, Sangdoo Yun, and Hwalsuk Lee. Character region awareness for text detection. In Computer Vision and Pattern Recognition, pages 9365–9374, 2019.
- [4] Bowen Baker, Ilge Akkaya, Peter Zhokov, Joost Huizinga, Jie Tang, Adrien Ecoffet, Brandon Houghton, Raul Sampedro, and Jeff Clune. Video pretraining (vpt): Learning to act by watching unlabeled online videos. In Advances in Neural Information Processing Systems, volume 35, pages 24639–24654, 2022.
- [5] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023.
- [6] Jake Bruce, Michael D Dennis, Ashley Edwards, Jack Parker-Holder, Yuge Shi, Edward Hughes, Matthew Lai, Aditi Mavalankar, Richie Steigerwald, Chris Apps, et al. Genie: Generative interactive environments. In International Conference on Machine Learning, 2024.
- [7] Haoxuan Che, Xuanhua He, Quande Liu, Cheng Jin, and Hao Chen. Gamegen-x: Interactive open-world game video generation. arXiv preprint arXiv:2411.00769, 2024.
- [8] Gang Dai, Yifan Zhang, Quhui Ke, Qiangya Guo, and Shuangping Huang. One-dm: One-shot diffusion mimicker for handwritten text generation. In European Conference on Computer Vision, pages 410–427. Springer, 2024.
- [9] Decart. Oasis: A universe in a transformer. 2024.
- [10] Haoyi Duan, Hong-Xing Yu, Sirui Chen, Li Fei-Fei, and Jiajun Wu. Worldscore: A unified evaluation benchmark for world generation. arXiv preprint arXiv:2504.00983, 2025.
- [11] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In International Conference on Machine Learning, 2024.
- [12] Linxi Fan, Guanzhi Wang, Yunfan Jiang, Ajay Mandlekar, Yuncong Yang, Haoyi Zhu, Andrew Tang, De-An Huang, Yuke Zhu, and Anima Anandkumar. Minedojo: Building open-ended embodied agents with internet-scale knowledge. In Advances in Neural Information Processing Systems, volume 35, pages 18343–18362, 2022.
- [13] Yuming Fang, Hanwei Zhu, Yan Zeng, Kede Ma, and Zhou Wang. Perceptual quality assessment of smartphone photography. In Computer Vision and Pattern Recognition, 2020.
- [14] Ruili Feng, Han Zhang, Zhantao Yang, Jie Xiao, Zhilei Shu, Zhiheng Liu, Andy Zheng, Yukun Huang, Yu Liu, and Hongyang Zhang. The matrix: Infinite-horizon world generation with real-time moving control. arXiv preprint arXiv:2412.03568, 2024.
- [15] FFmpeg Team. FFmpeg: A complete, cross-platform solution to record, convert and stream audio and video. https://ffmpeg.org/, 2024.
- [16] Shenyuan Gao, Jiazhi Yang, Li Chen, Kashyap Chitta, Yihang Qiu, Andreas Geiger, Jun Zhang, and Hongyang Li. Vista: A generalizable driving world model with high fidelity and versatile controllability. arXiv preprint arXiv:2405.17398, 2024.
- [17] Yuchao Gu, weijia Mao, and Mike Zheng Shou. Long-context autoregressive video modeling with next-frame prediction. arXiv preprint arXiv:2503.19325, 2025.

- [18] Junliang Guo, Yang Ye, Tianyu He, Haoyu Wu, Yushu Jiang, Tim Pearce, and Jiang Bian. Mineworld: a real-time and open-source interactive world model on minecraft. arXiv preprint arXiv:2504.08388, 2025.
- [19] Tarun Gupta, Wenbo Gong, Chao Ma, Nick Pawlowski, Agrin Hilmkil, Meyer Scetbon, Marc Rigter, Ade Famoti, Ashley Juan Llorens, Jianfeng Gao, et al. The essential role of causality in foundation world models for embodied ai. arXiv preprint arXiv:2402.06665, 2024.
- [20] David Ha and Jürgen Schmidhuber. World models. arXiv preprint arXiv:1803.10122, 2018.
- [21] Danijar Hafner. Embodied Intelligence Through World Models. PhD thesis, University of Toronto (Canada), 2024.
- [22] Hao He, Yinghao Xu, Yuwei Guo, Gordon Wetzstein, Bo Dai, Hongsheng Li, and Ceyuan Yang. Cameractrl: Enabling camera control for text-to-video generation. arXiv preprint arXiv:2404.02101, 2024.
- [23] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In Advances in Neural Information Processing Systems, volume 33, pages 6840–6851, 2020.
- [24] Anthony Hu, Lloyd Russell, Hudson Yeo, Zak Murez, George Fedoseev, Alex Kendall, Jamie Shotton, and Gianluca Corrado. Gaia-1: A generative world model for autonomous driving. arXiv preprint arXiv:2309.17080, 2023.
- [25] Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, et al. Vbench: Comprehensive benchmark suite for video generative models. In Computer Vision and Pattern Recognition, pages 21807–21818, 2024.
- [26] Junjie Ke, Qifei Wang, Yilin Wang, Peyman Milanfar, and Feng Yang. MUSIQ: multi-scale image quality transformer. arXiv preprint arXiv:2108.05997, 2021.
- [27] Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan Foster, Grace Lam, Pannag Sanketi, et al. Openvla: An open-source vision-language-action model. arXiv preprint arXiv:2406.09246, 2024.
- [28] Diederik P Kingma, Max Welling, et al. Auto-encoding variational bayes, 2013.
- [29] Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, et al. Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603, 2024.
- [30] Black Forest Labs. Flux. https://github.com/black-forest-labs/flux, 2024.
- [31] LAION-AI. aesthetic-predictor. https://github.com/LAION-AI/aesthetic-predictor, 2022.
- [32] Zhen Li, Zuo-Liang Zhu, Ling-Hao Han, Qibin Hou, Chun-Le Guo, and Ming-Ming Cheng. Amt: All-pairs multi-field transforms for efficient frame interpolation. In Computer Vision and Pattern Recognition, 2023.
- [33] Bin Lin, Yunyang Ge, Xinhua Cheng, Zongjian Li, Bin Zhu, Shaodong Wang, Xianyi He, Yang Ye, Shenghai Yuan, Liuhan Chen, et al. Open-sora plan: Open-source large video generation model. arXiv preprint arXiv:2412.00131, 2024.
- [34] Hongbin Lin, Zilu Guo, Yifan Zhang, Shuaicheng Niu, Yafeng Li, Ruimao Zhang, Shuguang Cui, and Zhen Li. Drivegen: Generalized and robust 3d detection in driving via controllable text-to-image diffusion generation. In Computer Vision and Pattern Recognition, 2025.
- [35] Hao Liu, Wilson Yan, Matei Zaharia, and Pieter Abbeel. World model on million-length video and language with ringattention. arXiv e-prints, pages arXiv–2402, 2024.
- [36] Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. In International Conference on Learning Representations, 2023.
- [37] Yueen Ma, Zixing Song, Yuzheng Zhuang, Jianye Hao, and Irwin King. A survey on vision-language-action models for embodied ai. arXiv preprint arXiv:2405.14093, 2024.
- [38] MineRL Project. MineRL: A large-scale dataset of minecraft demonstrations. https://minerl. readthedocs.io/en/latest/, 2020.
- [39] Haomiao Ni, Changhao Shi, Kai Li, Sharon X Huang, and Martin Renqiang Min. Conditional image-tovideo generation with latent flow diffusion models. In Computer Vision and Pattern Recognition, pages 18444–18455, 2023.
- [40] OpenAI. Sora: Video generation models as world simulators. https://openai.com/index/ video-generation-models-as-world-simulators/, 2024.
- [41] J Parker-Holder, P Ball, J Bruce, V Dasagi, K Holsheimer, C Kaplanis, A Moufarek, G Scully, J Shar, J Shi, et al. Genie 2: A large-scale foundation world model. URL: https://deepmind. google/discover/blog/genie2-a-large-scale-foundation-world-model, 2024.
- [42] William Peebles and Saining Xie. Scalable diffusion models with transformers. In International Conference on Computer Vision, pages 4195–4205, 2023.

- [43] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe Penna, and Robin Rombach. SDXL: Improving latent diffusion models for high-resolution image synthesis. In International Conference on Learning Representations, 2023.
- [44] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International Conference on Machine Learning, 2021.
- [45] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In Computer Vision and Pattern Recognition, pages 10684– 10695, 2022.
- [46] Sefik Serengil and Alper Ozpinar. A benchmark of facial recognition pipelines and co-usability performances of modules. Journal of Information Technologies, 17(2):95–107, 2024.
- [47] Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In International Conference on Machine Learning, pages 2256–2265. PMLR, 2015.
- [48] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. In International Conference on Learning Representations, 2021.
- [49] Tomás Soucek and Jakub Lokoc. Transnet v2: An effective deep network architecture for fast shot transition detection. In ACM International Conference on Multimedia, pages 11218–11221, 2024.
- [50] Zachary Teed and Jia Deng. Droid-slam: Deep visual slam for monocular, stereo, and rgb-d cameras. In Advances in Neural Information Processing Systems, volume 34, pages 16558–16569, 2021.
- [51] Linrui Tian, Qi Wang, Bang Zhang, and Liefeng Bo. Emo: Emote portrait alive generating expressive portrait videos with audio2video diffusion model under weak conditions. In European Conference on Computer Vision, pages 244–260. Springer, 2024.
- [52] Dani Valevski, Yaniv Leviathan, Moab Arar, and Shlomi Fruchter. Diffusion models are real-time game engines. arXiv preprint arXiv:2408.14837, 2024.
- [53] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in Neural Information Processing Systems, volume 30, 2017.
- [54] Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, Jianyuan Zeng, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025.
- [55] Xiaofeng Wang, Zheng Zhu, Guan Huang, Xinze Chen, Jiagang Zhu, and Jiwen Lu. Drivedreamer: Towards real-world-drive world models for autonomous driving. In European Conference on Computer Vision, pages 55–72, 2024.
- [56] Zhouxia Wang, Ziyang Yuan, Xintao Wang, Yaowei Li, Tianshui Chen, Menghan Xia, Ping Luo, and Ying Shan. Motionctrl: A unified and flexible motion controller for video generation. In ACM SIGGRAPH, pages 1–11, 2024.
- [57] World Labs. Generating worlds. https://www.worldlabs.ai/blog, 2025.
- [58] Haoning Wu, Erli Zhang, Liang Liao, Chaofeng Chen, Jingwen Hou Hou, Annan Wang, Wenxiu Sun Sun, Qiong Yan, and Weisi Lin. Exploring video quality assessment on user generated contents from aesthetic and technical perspectives. In International Conference on Computer Vision, 2023.
- [59] Zeqi Xiao, Yushi Lan, Yifan Zhou, Wenqi Ouyang, Shuai Yang, Yanhong Zeng, and Xingang Pan. Worldmem: Long-term consistent world simulation with memory. arXiv preprint arXiv:2504.12369, 2025.
- [60] Haofei Xu, Jing Zhang, Jianfei Cai, Hamid Rezatofighi, and Dacheng Tao. Gmflow: Learning optical flow via global matching. In Computer Vision and Pattern Recognition, pages 8121–8130, 2022.
- [61] Mingyu Yang, Junyou Li, Zhongbin Fang, Sheng Chen, Yangbin Yu, Qiang Fu, Wei Yang, and Deheng Ye. Playable game generation. arXiv preprint arXiv:2412.00887, 2024.
- [62] Sherry Yang, Yilun Du, Seyed Kamyar Seyed Ghasemipour, Jonathan Tompson, Leslie Pack Kaelbling, Dale Schuurmans, and Pieter Abbeel. Learning interactive real-world simulators. In International Conference on Learning Representations, 2024.
- [63] Sherry Yang, Jacob C Walker, Jack Parker-Holder, Yilun Du, Jake Bruce, Andre Barreto, Pieter Abbeel, and Dale Schuurmans. Position: video as the new language for real-world decision making. In International Conference on Machine Learning, 2024.
- [64] Shiyuan Yang, Liang Hou, Haibin Huang, Chongyang Ma, Pengfei Wan, Di Zhang, Xiaodong Chen, and Jing Liao. Direct-a-video: Customized video generation with user-directed camera movement and object motion. In ACM SIGGRAPH, pages 1–12, 2024.

- [65] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video diffusion models with an expert transformer. arXiv preprint arXiv:2408.06072, 2024.
- [66] Jiwen Yu, Yiran Qin, Haoxuan Che, Quande Liu, Xintao Wang, Pengfei Wan, Di Zhang, and Xihui Liu. Position: Interactive generative video as next-generation game engine. arXiv preprint arXiv:2503.17359, 2025.
- [67] Jiwen Yu, Yiran Qin, Xintao Wang, Pengfei Wan, Di Zhang, and Xihui Liu. Gamefactory: Creating new games with generative interactive videos. arXiv preprint arXiv:2501.08325, 2025.
- [68] Lijun Yu, José Lezama, Nitesh B Gundavarapu, Luca Versari, Kihyuk Sohn, David Minnen, Yong Cheng, Vighnesh Birodkar, Agrim Gupta, Xiuye Gu, et al. Language model beats diffusion–tokenizer is key to visual generation. In International Conference on Learning Representations, 2024.
- [69] Yifan Zhang and Bryan Hooi. Hipa: enabling one-step text-to-image diffusion models via high-frequencypromoting adaptation. arXiv preprint arXiv:2311.18158, 2023.
- [70] Yifan Zhang, Bryan Hooi, Lanqing Hong, and Jiashi Feng. Self-supervised aggregation of diverse experts for test-agnostic long-tailed recognition. In Advances in Neural Information Processing Systems, volume 35, pages 34077–34090, 2022.
- [71] Yifan Zhang, Bingyi Kang, Bryan Hooi, Shuicheng Yan, and Jiashi Feng. Deep long-tailed learning: A survey. IEEE Transactions on Pattern Analysis and Machine Intelligence, 45(9):10795–10816, 2023.
- [72] Yifan Zhang, Daquan Zhou, Bryan Hooi, Kai Wang, and Jiashi Feng. Expanding small-scale datasets with guided imagination. In Advances in Neural Information Processing Systems, volume 36, pages 76558–76618, 2023.
- [73] Longtao Zheng, Yifan Zhang, Hanzhong Guo, Jiachun Pan, Zhenxiong Tan, Jiahao Lu, Chuanxin Tang, Bo An, and Shuicheng Yan. Memo: Memory-guided diffusion for expressive talking video generation. arXiv preprint arXiv:2412.04448, 2024.
- [74] Zheng Zhu, Xiaofeng Wang, Wangbo Zhao, Chen Min, Nianchen Deng, Min Dou, Yuqi Wang, Botian Shi, Kai Wang, Chi Zhang, et al. Is sora a world simulator? a comprehensive survey on general world models and beyond. arXiv preprint arXiv:2405.03520, 2024.

- (a) Camera ↑

[Figure 494]

- (b) Camera ↓

[Figure 495]

- (c) Camera ←

[Figure 496]

- (d) Camera →

[Figure 497]

- (e) Camera ↖

[Figure 498]

- (f) Camera ↗

[Figure 499]

- (g) Camera ↙

[Figure 500]

- (h) Camera ↘

- Figure 12: Demonstration of Matrix-Game’s controllable video generation conditioned on diverse mouse movement commands, including horizontal (left, right), vertical (up, down), and diagonal camera adjustments. The model responds accurately to subtle changes in camera direction and produces geometrically consistent world views aligned with user control signals.

[Figure 501]

- (a) Forward

[Figure 502]

- (b) Back

[Figure 503]

- (c) Left

[Figure 504]

- (d) Right

[Figure 505]

- (e) Attack

[Figure 506]

- (f) Jump

- Figure 13: Demonstration of Matrix-Game’s controllable video generation conditioned on various keyboard actions, including forward, back, left, right, jump, and attack. The model accurately responds to user’s control signals and generates coherent motion patterns.

- (a) Forward + Left

[Figure 508]

- (b) Forward + Right

[Figure 509]

- (c) Backward + Left

[Figure 510]

- (d) Backward + Right

[Figure 511]

(e) Jump + Attack

[Figure 512]

(f) Backward + Attack

[Figure 513]

- (g) Forward + Jump

[Figure 514]

- (h) Forward + Attack

- Figure 14: Demonstration of Matrix-Game’s controllable video generation conditioned on complex actions, such as forward + right, backward + attack and jump + attack,. The model effectively interprets complex user commands and generates coherent, action-consistent motion trajectories that reflect the intended behaviors.

[Figure 515]

(a) Village

[Figure 516]

(b) Hall

[Figure 517]

- (c) Oriental Palace

[Figure 518]

- (d) Oriental Palace
- (e) Stadium

[Figure 519]

- (f) City

[Figure 520]

- (g) City

[Figure 521]

[Figure 522]

(h) Ancient City Wall

- Figure 15: Controllable world generation results of Matrix-Game across distinct Unreal scenarios. These demos illustrate the model’s ability to handle diverse environments, ranging from Village, City, and Stadium to Oriental Palace and Hall.

