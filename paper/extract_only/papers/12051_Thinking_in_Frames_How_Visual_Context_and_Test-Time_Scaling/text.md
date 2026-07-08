## Thinking in Frames: How Visual Context and Test-Time Scaling Empower Video Reasoning

Chengzu Li*12 Zanyi Wang*3 Jiaang Li*2 Yi Xu1 Han Zhou1 Huanyu Zhang4 Ruichuan An5 Dengyang Jiang6 Zhaochong An2 Ivan Vuli´c1 Serge Belongie2 Anna Korhonen1

# arXiv:2601.21037v1[cs.LG]28Jan2026

### Abstract

Vision-Language Models have excelled at textual reasoning, but they often struggle with finegrained spatial understanding and continuous action planning, failing to simulate the dynamics required for complex visual reasoning. In this work, we formulate visual reasoning by means of video generation models, positing that generated frames can act as intermediate reasoning steps between initial states and solutions. We evaluate their capacity in two distinct regimes: MAZENAVIGATION for sequential discrete planning with low visual change and TANGRAMPUZZLE for continuous manipulation with high visual change. Our experiments reveal three critical insights: (1) Robust Zero-Shot Generalization: In both tasks, the model demonstrates strong performance on unseen data distributions without specific finetuning. (2) Visual Context: The model effectively uses visual context as explicit control, such as agent icons and tangram shapes, enabling it to maintain high visual consistency and adapt its planning capability robustly to unseen patterns. (3) Visual Test-Time Scaling: We observe a test-time scaling law in sequential planning; increasing the generated video length (visual inference budget) empowers better zero-shot generalization to spatially and temporally complex paths. These findings suggest that video generation is not merely a media tool, but a scalable, generalizable paradigm for visual reasoning.

### 1. Introduction

The ability for reasoning and planning, decomposing a complex goal into actionable steps, has been approached through

*Equal contribution 1University of Cambridge 2Pioneer Center for AI, University of Copenhagen 3University of California San Diego 4Institute of Automation, Chinese Academy of Sciences 5Peking University 6Hong Kong University of Science and Technology. Correspondence to: Chengzu Li <cl917@cam.ac.uk>.

Preprint. January 30, 2026.

Large Language Models (LLMs) (Achiam et al., 2023; Yang et al., 2025a; Comanici et al., 2025). However, while Multimodal Large Language Models (MLLMs) have achieved remarkable success in understanding and reasoning over visual semantics (Bai et al., 2025a;b), they face significant limitations in embodied and spatial domains for reasoning (Li et al., 2024; Yang et al., 2025c). One of the bottlenecks lies in the medium of reasoning modality: MLLMs typically reason via textual descriptions, which are often inefficient and imprecise for capturing fine-grained geometric contexts and physical dynamics (Gao et al., 2025; Li et al., 2025c; Fu et al., 2025) (e.g., describing the exact continuous rotation and placement of a tangram piece or a collision-free trajectory in an irregular maze).

Recognizing these limitations, recent research has shifted toward reasoning with multimodal traces, such as MVoT (Li et al., 2025d) and Visual Planning (Xu et al., 2025), proposing that thinking directly in the visual domain provides a higher-bandwidth representation for spatial intelligence. Concurrently, video generation models have been viewed primarily as tools for media creation, optimized for aesthetic quality rather than logical consistency (Huang et al., 2024; Wan et al., 2025). While recent proprietary models like Veo 3 (Google, 2025) have qualitatively explored using video for reasoning, these efforts lack in-depth quantitative investigation and open reproducibility (Wiedemer et al., 2025). We argue that video generation offers a unique paradigm for reasoning: by operating on a dense temporal manifold, the model acts as a continuous proxy of the reasoning process. This offers better interpretability and verifiability compared to single-step image editing (Cai et al., 2025; Tong et al., 2026), as it captures the process of change.

Despite the promise of visual expressiveness, employing generative video for planning introduces inherent challenges regarding controllability and fidelity. Prior work in visual reasoning (Xu et al., 2025; Guo et al., 2025; Tong et al., 2025b) has largely focused on tasks with discrete actions such as maze navigation, where visual changes between frames are minimal. While these studies establish a foundation, they have been confined to in-distribution settings, for instance, evaluating models on familiar grid sizes and

[Figure 1]

- Figure 1. Video generation models as visual reasoners, empowered by (1) enriched visual context for improved geometric control and

(2) visual test-time scaling that allocates a larger inference-frame budget and enables stronger performance on long-horizon, complex sequential planning tasks, together demonstrating robust generalization across diverse scenarios.

patterns for maze navigation (Xu et al., 2025; He et al., 2025). A key aspect of reasoning, however, also involves generalization (Huang et al., 2025): the ability to apply learned rules, such as collision avoidance or motion control, to unseen difficulty levels (e.g., larger mazes) and novel visual contexts (e.g., unseen agent avatars or object shapes). Furthermore, prior work doesn’t fully evaluate models’ capabilities in controllability and fidelity as generative visual reasoners. Maintaining complex geometric consistency under continuous action spaces with high visual changes remains challenging for generative models (Wang et al., 2025; Wu et al., 2025a), and existing studies obscure these difficulties by focusing on simpler or highly constrained settings.

In this work, we conduct an in-depth investigation of video generation models for visual planning, aiming to understand whether and how these models support planning under diverse challenges. To cover the spectrum of visual reasoning, we evaluate two distinct regimes: (1) MAZENAVIGATION representing sequential, structured planning with low visual change and (2) TANGRAMPUZZLE for continuous manipulation with high visual change. Unlike prior work, MAZENAVIGATION is investigated with various outof-distribution settings, and TANGRAMPUZZLE forces the model to handle dynamic visuals, requiring precise geometric preservation during continuous manipulation.

Our experiments reveal that video-based reasoning consistently outperforms text-based baselines, particularly in TANGRAMPUZZLE where MLLMs struggle to verbalize precise spatial manipulations. Beyond raw performance, we observe two critical emergent properties. First, the model demonstrates Visual Context as Control: by conditioning on specific visual context such as agent icons or puzzle shapes, the model maintains high visual consistency and robustly adapts to unseen patterns without fine-tuning. This allows for strong zero-shot performance across spatially and

visually Out-Of-Distribution (OOD) settings. Second, we uncover a Visual Test-Time Scaling Law in sequential planning that parallels “System 2” thinking in human cognition (Kahneman, 2011). We observe that increasing the generated video frames, which effectively increases the visual inference budget, empowers the model to solve spatially and temporally complex paths that fail at lower frame counts (e.g., extending the video length from 81 to 121 frames). We posit that in the visual domain, temporal frames can also act as a compute budget for reasoning, allowing the model to “think” for longer about the solution.

The main contributions of this paper include:

- • New Regime for Visual Reasoning: In addition to OOD situations for MAZENAVIGATION, we introduce TANGRAMPUZZLE, which shifts spatial reasoning from static, discrete planning to a continuous dynamic process capable of handling high visual change with geometric constraints.
- • Visual Context for Robust Generalization: We demonstrate that video generation models can abstract underlying planning algorithms rather than memorizing pixels with OOD visuals and maintain geometric consistency by offering visual context as control.
- • Visual Test-Time Scaling: We identify a scaling law in sequential planning, showing that increasing the inference frame budget acts as test-time compute, significantly improving generalization on long-horizon, complex tasks.

### 2. Related Work

Multimodal Planning. To extend textual reasoning traces to multimodality, one line of research employs MLLMs and grounds the reasoning process through interleaved textimage sequences generated by external tools. These tools include symbolic programming languages used as sketchpads (Hu et al., 2024), explicit coordinate-based representations

to enhance perceptual grounding (Su et al., 2025; Fan et al., 2025), and other generative models invoked via tool calls (Zheng et al., 2025; Cheng et al., 2026). Beyond tool-based approaches, another line of work focuses on native multimodal visual planning by designing MLLM architectures capable of directly generating multimodal content. This includes image sequence (Xu et al., 2025) and interleaved multimodal traces generation (Li et al., 2025d), as well as recursive mixed-modality generation (Li et al., 2025a; Gu et al., 2025). A separate direction leverages latent representations for reasoning; however, these methods generally lack explicit visualized “visual thoughts” (Li et al., 2025b; Tong et al., 2025a) except for Zhang et al. (2025b). Despite these advances, prior work has largely focused on in-distribution settings and structured planning tasks, with limited investigation into out-of-distribution (OOD) generalization. In addition to interleaved multimodal traces, recent studies have also explored video-based representations for visual planning, which we discuss in the following paragraph.

Video Generation. Video generation models have traditionally been studied primarily as media creation tools, with an emphasis on visual fidelity, temporal coherence, and consistency as world models (Huang et al., 2024; Wan et al., 2025; Team et al., 2025). More recently, efforts have begun to repurpose video generation models as visual reasoners, with qualitative evaluations conducted on both proprietary models (Wiedemer et al., 2025) and open-source models, either in zero-shot settings (Guo et al., 2025) or after finetuning (Yang et al., 2025b). However, these studies largely focus on maze navigation, a form of sequential discrete action planning that involves minimal visual change across frames while preserving a fixed spatial layout. Beyond their emphasis on in-distribution performance, the fundamental challenges faced by generative visual reasoners remain underexplored, particularly scenarios involving continuous manipulation with substantial visual changes over time, where maintaining geometric and spatial consistency throughout the video remains a significant challenge (Fan et al., 2024).

Test-Time Scaling. Chain-of-Thought (CoT) prompting (Wei et al., 2022b) enables models to solve complex problems by explicitly generating intermediate reasoning steps. Recent work (OpenAI, 2024) has shown that scaling testtime compute, i.e., increasing computation during inference either through multiple parallel sampled inferences with aggregation (Brown et al., 2025) or a single inference with longer traces (Muennighoff et al., 2025), can yield performance improvements analogous to scaling training data. In the visual domain, however, such test-time scaling has not been systematically explored for video generation models. Liu et al. (2025) investigated parallel test-time scaling for video generation, targeting higher perceptual quality rather than visual planning or reasoning. In contrast, we are the first to identify a direct visual analogue of test-time scaling

laws for planning. Specifically, we show that increasing the frame budget in video generation is empirically similar to increasing the length of reasoning traces in LLMs, enabling models to solve planning problems with OOD complexity.

### 3. Video Generation for Visual Planning

#### 3.1. Formulation

Following the definition from Xu et al. (2025), we formally frame video generation as a visual planning problem. Given an initial state image sstart ∈ S, a goal specification g (e.g., a target image or pattern), and a set of latent physical constraints c, the objective is to generate a video sequence V = {v0,v1,...,vT} where v0 = sstart. A valid solution trajectory must transition from sstart to a final state vT that satisfies the goal condition g, while maintaining temporal consistency and adhering to constraints c (e.g., collision avoidance, object permanence). In this framework, the generative model Pθ(V |sstart,g) acts as the planning policy, where the temporal evolution of frames vt → vt+1 corresponds to the execution of the plan. Unlike symbolic planners that output discrete actions at, the video model outputs continuous high-dimensional dense transitions, requiring the implicit learning of latent rules and causal dynamics.

#### 3.2. Spectrum of Visual Planning Regimes

Prior research has predominantly focused on grid-world navigation (e.g., mazes) (Xu et al., 2025; Guo et al., 2025). While effective for testing sequential logic, these environments are visually static with local visual change: only a small agent moves against a fixed background. Such tasks operate in discrete action spaces that can often be solved via symbolic proxies (Dao & Vu, 2025), overlooking the unique advantage of generative video models: the ability to model complex visual dynamics in continuous action spaces.

To rigorously evaluate the model’s capabilities, we define two contrasting reasoning regimes (summarized in Table 1). In addition to MAZENAVIGATION, TANGRAMPUZZLE introduces a novel stress test for visual planning. It requires the model to manage high visual change with geometric consistency, and continuous object manipulation, where text-based reasoning fundamentally fails.

MAZENAVIGATION. We adopt standard maze navigation to maintain comparability with previous work (Xu et al., 2025; Wu et al., 2025c), where the agent icon moves along the maze to get the final destination. This task evaluates the model’s ability to retain long-term consistency of the map structure and execute precise, collision-free pathfinding with minimal visual change of the agent along the path.

TANGRAMPUZZLE. We introduce TANGRAMPUZZLE as a novel challenge for continuous spatial manipulation. The

Table 1. Comparison of Reasoning Regimes.

Aspect MAZENAVIGATION TANGRAMPUZZLE

Visual Volatility Low High Action Space Discrete Continuous Core Challenge Long-horizon Logic Spatial Geometry

objective is to manipulate a set of 7 disjoint geometric pieces from the tangram to precisely fill a target silhouette. Unlike mazes, the difficulty for the visual generative model here is not path length but geometric preservation, since the length is fixed by the number of pieces. Throughout the reasoning process, the entire scene changes as pieces are translated and rotated continuously. Formally, given a target silhouette without any infilling pieces sstart, the model generates a solution vT (the final frame if using video generation) where all pieces pi are placed within g without overlapping or distortion as constraints c. This task presents a visual-native challenge where textual descriptions struggle to efficiently capture the precise relative locations and rotations required for a valid solution.

### 4. Experiments

We first describe the experimental setups (§4.1) and analyze the performance of different systems on both tasks, considering not only in-distribution results (§4.2) but also zero-shot generalization in out-of-distribution settings (§4.3).

#### 4.1. Experimental Setups

Data. We construct the MAZENAVIGATION dataset following Xu et al. (2025) and Wu et al. (2025c). The dataset covers varying levels of difficulty in both maze size (from 3 × 3 to 6 × 6) and visual appearance, including 40 distinct agent icons with different colors as shown in Figure 4 in Appendix A.1. For each data, the optimal navigation path from the start location to the goal is determined using heuristic search algorithms (Ivanitskiy et al., 2023). For TANGRAMPUZZLE, we build upon the dataset introduced by Ji et al. (2022), which evaluates whether MLLMs can recognize abstract patterns composed of tangram pieces. We repurpose this dataset into a sequential setting, where tangram pieces are placed into their target locations in the abstract pattern step by step. Additional details on data statistics, collection and preprocessing are provided in Appendix A.1.

Metrics. For MAZENAVIGATION, we adopt the same metrics from Xu et al. (2025) with Exact Match (EM) and Progress Rate (PR). For TANGRAMPUZZLE, evaluating generative fidelity requires strictly enforcing geometric constraints. We introduce a hierarchical evaluation suite assessed on the final generated frame vT using a pixel-level heuristic parser (details in Appendix A.3):

- 1. Strict Goal Completion: A trial is successful if and only if all N pieces are placed within the target silhouette without overlap, distortion, or color hallucination. This measures strict constraint satisfaction:

Sstrict =

N

i=1

I(pi ∈ g ∧ consistent(pi))

, where consistent(pi) ensures the piece retains its original topology (shape and color).

- 2. Progress Goal Completion (Piece-wise Accuracy): To measure partial success, we compute the normalized count of correctly placed pieces within the silhouette.

Sprogress =

1 N

N

i=1

I(pi ∈ g ∧ consistent(pi))

- 3. Boundary Adherence (IoU): We quantify the precision of the placement by calculating the intersection over union of the generated pieces with the target silhouette area.

SIoU =

Area(Pgen ∩ Ptarget) Area(Pgen ∪ Ptarget)

A score of 100 in percentage indicates perfect containment. Scores smaller than 100 indicate unfilled gaps or hallucinated pixels outside the boundary.

Models and Experiments. We adopt the Wan 2.2 TI2V 5B model (Wan et al., 2025) as the primary backbone for video generation. Wan 2.2 TI2V 5B is a strong text-to-video diffusion model conditional on textual instruction, which is pre-trained on large-scale video data with sequences of 81 frames. We fine-tune only a subset of the model parameters using LoRA (Hu et al., 2022) for 20 epochs. The computational cost and training details are reported in Appendix A.

In addition, we evaluate several strong proprietary and opensource MLLMs with promising visual understanding performance under zero-shot or fine-tuning settings, including GPT 5.1 (OpenAI, 2025a), GPT 5.2 (OpenAI, 2025b), and Qwen3-VL-8B (Bai et al., 2025a). For Qwen3-VL-8B, we apply full-parameter fine-tuning to support textual reasoning baselines. For visual reasoning, we compare against VPRL (Xu et al., 2025) on MAZENAVIGATION and Qwen-ImageEdit (Wu et al., 2025b) on TANGRAMPUZZLE, which serve as strong task-specific baselines. All fine-tuned models are optimized using standard training objectives without taskspecific auxiliary losses. Detailed hyperparameter configurations and prompting templates for each task and system variant are provided in Appendix A.2 and C.

- 4.2. In-Distribution Results

We first establish the efficacy of the proposed approach on in-distribution (IID) data, where test samples share the same distribution (maze size, icon, silhouette) as the training set.

- Table 2. Results across different maze sizes and path lengths. All open-sourced models are fine-tuned and proprietary models are evaluated in a zero-shot setting. represents texts, represents images, and represents videos. ■ represents visual reasoning systems.

In Distribution OOD Maze Sizes OOD Path Length OOD Both

3x3 – 6x6 7x7 8x8 5x5 (Long) 6x6 (Long) 7x7 (Long) 8x8 (Long)

Model Input Output

EM ↑ PR ↑ EM ↑ PR ↑ EM ↑ PR ↑ EM ↑ PR ↑ EM ↑ PR ↑ EM ↑ PR ↑ EM ↑ PR ↑ Proprietary Models

- GPT-5.1 + 10.6 10.7 6.32 6.72 6.00 6.00 0 0 0 0 0 0 0 0
- GPT-5.2 + 12.5 12.5 8.40 8.40 8.40 8.40 0 0 0 0 0 0 0 0 Open-Sourced Models (All Fine-Tuned) Qwen3-VL-8B + 58.3 68.6 20.0 37.3 19.2 34.3 0 13.3 0 13.2 0 11.3 0 8.9

- w coordinates 72.0 77.3 33.2 45.0 22.0 30.5 0 17.1 0 13.4 0 8.1 0 5.9

VPRL-7B * 73.5 78.6 14.0 25.2 4.00 6.20 0 11.0 2.00 16.7 0 4.10 0 0.70 Wan2.2-TI2V-5B + 96.0 99.0 90.0 92.3 80.0 83.6 44.0 55.2 42.0 51.6 40.0 51.1 32.0 47.1

###### - Unseen Visual Icons 95.5 98.2 92.0 92.6 78.0 81.6 36.0 46.3 42.0 52.0 38.0 47.9 32.0 42.3

Fade-InQwenImageEdit2509 Wan2.2TI2V5B Tangram Puzzle

[Figure 2]

[Figure 3]

[Figure 4]

✅

[Figure 5]

❌ ❌ ❌

Hard to Maintain Geometric Consistency

Rotation

[Figure 6]

[Figure 7]

[Figure 8]

Qwen VL (Text) Nano Banana Qwen Image Edit 2509 Wan 2.2 TI2V 5B

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

✅

✅

❌ ❌ ❌ ❌ ❌ ❌

Translation

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

Qwen VL (Text) Nano Banana Qwen Image Edit 2509 Wan 2.2 TI2V 5B

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

❌ ❌ ❌ ❌ ❌ ✅

✅ ✅

- Figure 2. Generated solution of TANGRAMPUZZLE by different system variants. For Qwen-3-VL, we visualize the layout based on the predicted coordinates and rotations. We crop the main area for the predictions from image editing model and video generation model. For video generation model, we only select the last frame

- as illustration here. For full details, please refer to Figure 13.

MLLMs struggle with interpretable reasoning with interpretability. Proprietary MLLM consistently struggles with MAZENAVIGATION, suggesting that while visual reasoning is trivial for humans, it remains a bottleneck for current large-scale models in line with previous work (Xu et al., 2025). Fine-tuning open-sourced MLLM offers limited improvement in OOD settings, with models still failing to reliably capture the underlying spatial layouts, with or without explicit coordinate supervision. This weakness is most evident in TANGRAMPUZZLE, where text-based baselines struggle to express continuous spatial manipulations with precise coordinates, leading to overlapping pieces and near-zero Strict Goal Completion (as shown in Figure 2 with a standalone renderer). We hypothesize this stems from the bottleneck of grounding, where MLLMs typically ground attention to semantic objects (Zhang et al., 2025a; Izadi et al., 2025), whereas in our context, the model grounds coordinates to empty space inside a silhouette without any semantic anchors. In contrast, the video generation approach explicitly simulates the reasoning process as a vi-

sual trajectory. By rendering the intermediate steps rather than outputting abstract coordinates, our model circumvents cross-modal grounding (visual to symbolic), offering better performance and native interpretability of failure modes, such as visual collisions, that text-based planners obscure.

Video generation model enables reliable sequential planning. As detailed in Table 2, the video model exhibits strong consistency in IID MAZENAVIGATION. On standard 4 × 4 and 5 × 5 grids, the model attains 98.0% and 96.0% Exact Match (EM) scores respectively. These results indicate that within the training distribution, the model successfully learns to model valid spatiotemporal transitions, consistently generating trajectories that adhere to environmental constraints while successfully reaching the goal.

Visual context is essential for TANGRAMPUZZLE as geometric priors. For TANGRAMPUZZLE, we introduce three system variants that vary the availability of visual context and the action types in order to examine how geometric priors affect model performance (examples in Figure 5):

- • Fade-in: Tangram pieces gradually appear in their target locations on an initially empty canvas, one by one.
- • Rotation: Pieces initially are listed outside the target silhouette with random orientations on the left side of the canvas. The model must first rotate and then translate each piece to fit the target shape, requiring both geometric transformation and spatial planning.
- • Translation: Pieces start outside the silhouette on the left side of the canvas, but are pre-aligned with the correct orientation. The model is required only to translate pieces into position, isolating planning from rotational reasoning.

As shown in Table 3, performance differs markedly across these variants depending on the available visual context. In the Fade-In setting, where initial tangram shapes as geometric priors are absent, the model fails completely (0.8% accuracy), mirroring its inability to fit the training data with ∼ 0% training accuracy. Rotation achieves intermediate per-

- Table 3. Quantitative results under Fade-In, Rotation, Translation situations for TANGRAMPUZZLE. “GC.” denotes goal completion, “BA.” denotes boundary adherence. represents texts, represents images, and represents videos.■ represents visual reasoning systems.

Seen (Learnability) Unseen (Generalizability)

Model Input Output Strict GC Progress GC BA Strict GC Progress GC BA Fade-In Qwen-Image-Edit-20B + 31.0 82.3 99.8 32.0 81.3 99.7 Wan2.2-TI2V-5B + 0.80 49.4 98.1 0.80 48.9 98.0 Rotation Qwen3-VL-8B + 14.4 69.7 89.5 1.6 52.1 80.8 Nano Banana + - - - 9.80 43.4 64.7 Qwen-Image-Edit-20B + 45.2 87.5 99.7 43.2 85.7 99.6 Wan2.2-TI2V-5B + 22.4 76.8 98.1 22.4 74.5 98.0 Translation Qwen3-VL-8B + 28.0 75.7 91.4 1.60 58.9 82.4 Nano Banana + - - - 3.90 51.3 74.5 Qwen-Image-Edit-20B + 85.7 97.7 99.9 76.0 95.4 99.7 Wan2.2-TI2V-5B + 68.0 94.7 97.0 60.8 92.0 97.0

formance 22.4%, likely because while it maintains temporal context, the pixel-level warping during rotation degrades feature consistency. Conversely, when the visual context of the piece with more prior knowledge (shapes and orientations) is explicitly preserved via Translation, performance reaches 68.0%. This gap confirms that the presence of visual context helps model to learn geometric planning with better consistency, which is further discussed in Section 5.1.

- 4.3. Zero-Shot Out-of-Distribution Generalization

with novel, unseen patterns (OOD Icon). Remarkably, this visual shift induces negligible performance loss. On 3 × 3 mazes, accuracy shifts marginally from 94.00% with the seen icon to 92.00% with the OOD icon; on 5 × 5 mazes, it remains robust at 94.00%. This consistency holds across all OOD maze sizes and path lengths (Table 2, right columns). Qualitatively (Figure 11), the model successfully preserves the identity and texture of the unseen icon throughout the generation, adhering to the object permanence constraint. These results indicate that the model has decoupled the planning algorithm from the visual entity; it does not simply memorize specific pixel transitions, but rather identifies the agent as a distinct entity, and applies learned movement dynamics disentangled from its visual appearance.

A key criticism of visual agents is their tendency to overfit to visual assets. We demonstrate that our model generalizes to Out-Of-Distribution (OOD) contexts without fine-tuning.

Generalization to OOD Maze Sizes and Path Lengths. We first probe the model’s ability to handle unseen spatial scales. As detailed in Table 2, the model generalizes robustly to spatially larger grid dimensions. When transitioning from seen 6 × 6 grids to unseen 7 × 7 grids, EM accuracy remains high at 92.00%. While performance naturally degrades on 8 × 8 grids (78.00%), this represents a graceful decline rather than a catastrophic failure, suggesting the model’s planning horizon scales reasonably well with spatial expansion. However, we observe a sharper performance drop when the length of the path exceeds the training distribution (temporal OOD Path Length), with accuracy falling to 36 ∼ 42%. Crucially, rather than the reasoning capability itself, as we demonstrate in Section 5.2, this performance drop in either spatial or temporal OOD cases can be effectively mitigated via “Visual Test-Time Scaling” by increasing the generated frame count to allocate more compute for complex trajectories.

Geometric Generalization to OOD Tangram Silhouettes. In the Tangram domain, we test whether the model can solve puzzles with unseen target silhouettes. As shown in Table 3, the model exhibits strong zero-shot transfer. In the Translation setting, accuracy on unseen silhouettes (60.8%) is comparable to seen silhouettes (68.0%). Furthermore, even in the Rotation setting, changing the initial layout of the pieces (e.g., using the Translation test configuration) yields performance similar to the in-distribution layout. This parity between seen and unseen geometries confirms that the model is not retrieving memorized solutions. Instead, it has internalized generalized concepts of geometric fitting and collision-free sliding, allowing it to solve novel spatial arrangement problems dynamically.

### 5. Discussion and Analysis

#### 5.1. Visual Context as Control Constraints

Robustness for OOD Visual Agent Icon in MAZENAVIGATION. To determine if the model relies on and overfits to specific visual artifacts, we replace the standard agent icon

Rather than relying on textual instructions that are then mapped to visual patterns (Wan et al., 2025), we posit that

###### OOD Maze Size

###### OOD Path Length

###### OOD Both

105

90

###### EM PR

###### EM PR

###### EM PR

70

100

80

95

60

70

90

50

60

Score

85

50

40

80

40

75

30

30

70

20

65

20

61 81 101 121 141

61 81 101 121 141

61 81 101 121 141

Frames

Frames

Frames

90

65

EM PR

###### 80 EM PR

EM PR

60

85

55

80

70

50

75

45

Score

60

40

70

35

50

65

30

60

25

40

55

20

5 7 9 11

5 7 9 11

5 7 9 11

Factor

Factor

Factor

- Figure 3. Visual Test-Time Scaling for MAZENAVIGATION using unseen icon with more inference budget. Row 1 shows the performance curve when increasing the total number of frames per video; Row 2 shows the performance curve when changing the scaling factor κ to allocate a different number of frames per discrete step in the maze solution. Detailed results for both settings are shown in Figure 6 and 7.

explicit visual context, by directly specifying visual appearance, acts as a stronger form of control for visual reasoning.

Visual context outperforms textual instructions for OOD visual generalization. In MAZENAVIGATION, we compared two control methods for changing the agent’s appearance: (1) providing a visual of the new icon in the first frame

- at the starting point (as visual anchoring), vs. (2) providing a text description (e.g., “a blue star”). The results show that the model failed to consistently generate the correct agent via text prompts, often reverting to the training distribution’s default agent. Conversely, with visual context, the model successfully generalized to unseen icons while navigating unseen mazes. The model effectively learns a conditional policy P(Trajectory|Icon,Layout), treating the visual anchor as a variable to be preserved and the maze as a constraint to be navigated. This implies that, in lowresource settings with limited training data, visual context acts as a more natural and straightforward control signal than text, enabling zero-shot adaptation to new domains.

Visual context empowers geometric control for TANGRAMPUZZLE. Our analysis of TANGRAMPUZZLE reveals that visual context functions as a geometric control on the generation process. We observe a strict performance hierarchy dictated by whether the visual context is injected, as we introduced in Section 4.2. This hierarchy persists even when testing with baseline Image Editing models (using the first frame as input and the last frame as target), suggesting a consistent principle: visual context serves as a geometric reference for control. In the Translation setting, the visual context is the strongest because the visual input

provides a reference map where shape and orientation are preserved, requiring the model only to solve for location. This suggests that high-fidelity visual context anchors the reasoning process, allowing the model to robustly apply geometric transformations within a continuous generative dynamic rather than hallucinating new shapes.

#### 5.2. Visual Test-Time Scaling

Test-Time Compute, allowing a model to process information longer to achieve better results, has been investigated in LLMs (OpenAI, 2024; Muennighoff et al., 2025). We investigate if a parallel Visual Test-Time Scaling Law exists for video generation for visual planning by generating more frames act as a larger inference budget for reasoning.

Scaling inference budget improves OOD generalization. We find that increasing the total frame count (e.g., from 81 to 101, 121 frames) improves navigation performance on both spatially (maze size) and temporally (path length) OOD tasks. As shown in the first row in Figure 3, OOD performance increases steadily as we scale the inference budget from 61 to 121 frames. However, we observe a ceiling effect: when scaling to 141 frames for temporal OOD cases, performance drops compared to 121 frames, though it remains superior to the training baseline of 81 frames. We attribute this drop to the architectural limits of the video generation model’s positional embeddings (Wang et al., 2025), which struggle to extrapolate when the frame count deviates significantly from the training distribution.

To rigorously probe whether these gains stem from finer-

grained reasoning or simply longer video duration, we introduce a control variable κ (scaling factor), defined as the number of frames allocated per discrete step in the maze solution. We tested κ ∈ {5,7,9,11} across spatially and temporally OOD settings. As shown in Figure 3, we observe a clear positive correlation: assigning more frames per step (κ = 7,9,11) significantly improves performance on spatially ID and OOD settings compared to lower resolutions (κ = 5). Notably, in temporal OOD settings, performance peaks at κ = 9 before degrading at κ = 11. This degradation aligns with the positional embedding limitation noted above, as κ = 11 pushes the total video length to ∼ 200 frames. Crucially, this drop is not observed in the spatially OOD setting where total path length is in-distribution, confirming that the degradation is limited by the sequence length capacity rather than logical. These results confirm that, within architectural limits, the video generation model exhibits strong extrapolation capabilities, utilizing increased inference compute budget to resolve complex dependencies.

Emergent “self-correction” behaviors appear at higher frame budgets. Beyond statistical improvements, we qualitatively observe emergent behaviors of the agent that are not included in training data when the inference budget is increased. In some long-horizon maze cases (see Figure 8), the model initially steers the agent toward the wrong direction. However, unlike in low-frame regimes, where the agent would collide with the wall, the high-frame model generates a backtracking trajectory by stopping, reversing, and correcting its path within the generated sequence. This suggests that the video generation is not merely retrieving a memorized path, but actively simulating the trajectory, where intermediate frames help to correct the rollout plan.

Fidelity-Reasoning Trade-off: Scaling is task-dependent. While scaling benefits sequential planning in MAZENAVIGATION, we do not observe the same trend for TANGRAMPUZZLE. In the Translation setting, increasing the frame count maintains performance but yields no significant gain. To understand this discrepancy, we analyzed the correlation between an indicative Visual Consistency Metric (measuring shape integrity over time, detailed introduction in Appendix A.3) and task success in Rotation and Translation, finding a Pearson correlation of ρ ≥ 0.6. This indicates that one of the bottlenecks in TANGRAMPUZZLE is maintaining the shape’s geometry constraint during generation. Unlike the MAZENAVIGATION, which moves a small icon across a static environment, tangram pieces are susceptible to geometric deformation over long generation windows with high visual change across the whole canvas. Therefore, while more frames help logic in MAZENAVIGATION, they tax the model’s ability to maintain high-frequency visual details in TANGRAMPUZZLE, highlighting different bottlenecks for current generative visual reasoners.

#### 5.3. Discussion

Generalization to Irregular Maze and Movements Beyond generalizing to larger grids, we observe that the video generation model exhibits remarkable zero-shot adaptation to irregular maze layouts (Figure 9 in Appendix B). Despite being trained exclusively on grid-based environments with horizontal or vertical movement, the model successfully generates diagonal trajectories to navigate irregular, non-grid environments. While the success rate is lower compared to standard mazes, the model still shows its potential to adhere to collision constraints without hallucinating paths through walls, even in these unseen topologies. This indicates that the model has not simply memorized a discrete grid-based navigation, but has abstracted a policy of moving toward goal subject to collision constraints, which enables cross-domain adaptation, and we call for further research.

Video Generation v.s. Image Editing We observe that while Image Editing achieves higher performance on the TANGRAMPUZZLE task, Video Generation offers better interpretability and verifiability by producing the intermediate reasoning traces. We attribute the performance gap in Tangrams to two primary factors. In addition to the model sizes, where the image editing model is twice as large as the video generation model, another potential factor is optimization density. Video generation model requires modeling the joint probability of the entire sequence P(v0,...,vT), which requires more effort to fit every frames during training. In contrast, the image editing model optimizes only the conditional marginal P(sgoal|sstart), allowing for full capacity for optimizing final visual fidelity without being distracted by temporal consistency maintenance.

### 6. Conclusion

In this work, we demonstrate that video generation models are not merely for visual synthesis but also as powerful engines for visual reasoning. Through systematic investigations on the discrete logical challenges of MAZENAVIGATION and the continuous, high-fidelity geometric constraints of TANGRAMPUZZLE, we demonstrate that generative video affords an expressive representation for spatial planning that text-based MLLMs inherently lack. Our results show that video-based planners exhibit strong generalization, effectively disentangling task-level logic from superficial visual patterns. Beyond other performance bottlenecks, such as maintaining geometric consistency, we further identify a visual test-time scaling phenomenon: as the number of generated frames, serving as a temporal reasoning budget, increases, models display emergent competence on out-of-distribution, long-horizon sequential problems. This behavior mirrors the transition to powerful System-2style reasoning observed in Large Language Models, which we call for further research.

### References

Achiam, J., Adler, S., Agarwal, S., Ahmad, L., Akkaya, I., Aleman, F. L., Almeida, D., Altenschmidt, J., Altman, S., Anadkat, S., et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.

Bai, S., Cai, Y., Chen, R., Chen, K., Chen, X., Cheng, Z., Deng, L., Ding, W., Gao, C., Ge, C., Ge, W., Guo, Z., Huang, Q., Huang, J., Huang, F., Hui, B., Jiang, S., Li, Z., Li, M., Li, M., Li, K., Lin, Z., Lin, J., Liu, X., Liu, J., Liu, C., Liu, Y., Liu, D., Liu, S., Lu, D., Luo, R., Lv, C., Men, R., Meng, L., Ren, X., Ren, X., Song, S., Sun, Y., Tang, J., Tu, J., Wan, J., Wang, P., Wang, P., Wang, Q., Wang, Y., Xie, T., Xu, Y., Xu, H., Xu, J., Yang, Z., Yang, M., Yang, J., Yang, A., Yu, B., Zhang, F., Zhang, H., Zhang, X., Zheng, B., Zhong, H., Zhou, J., Zhou, F., Zhou, J., Zhu, Y., and Zhu, K. Qwen3-vl technical report, 2025a. URL https://arxiv.org/abs/2511.21631.

Bai, S., Chen, K., Liu, X., Wang, J., Ge, W., Song, S., Dang, K., Wang, P., Wang, S., Tang, J., Zhong, H., Zhu, Y., Yang, M., Li, Z., Wan, J., Wang, P., Ding, W., Fu, Z., Xu, Y., Ye, J., Zhang, X., Xie, T., Cheng, Z., Zhang, H., Yang, Z., Xu, H., and Lin, J. Qwen2.5-vl technical report, 2025b. URL https://arxiv.org/abs/2502.13923.

Brown, B., Juravsky, J., Ehrlich, R. S., Clark, R., Le, Q. V., Re, C., and Mirhoseini, A. Large language monkeys: Scaling inference compute with repeated sampling, 2025. URL https://openreview.net/forum? id=0xUEBQV54B.

Cai, H., Cao, S., Du, R., Gao, P., Hoi, S., Hou, Z., Huang, S., Jiang, D., Jin, X., Li, L., et al. Z-image: An efficient image generation foundation model with single-stream diffusion transformer. arXiv preprint arXiv:2511.22699, 2025.

Cheng, D., Li, Y., Ma, Z., Cai, H., Hu, Y., Wang, W., Nie, L., and Li, W. Omni-r1: Towards the unified generative paradigm for multimodal reasoning. arXiv preprint arXiv:2601.09536, 2026.

Comanici, G., Bieber, E., Schaekermann, M., Pasupat, I., Sachdeva, N., Dhillon, I., Blistein, M., Ram, O., Zhang, D., Rosen, E., et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025.

Dao, A. and Vu, D. B. Alphamaze: Enhancing large language models’ spatial intelligence via grpo. arXiv preprint arXiv:2502.14669, 2025.

Fan, J., Xue, H., Zhang, Q., and Chen, Y. Refdrop: Controllable consistency in image or video generation via

reference feature guidance. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024. URL https://openreview.net/forum? id=09nyBqSdUz.

Fan, Y., He, X., Yang, D., Zheng, K., Kuo, C.-C., Zheng, Y., Guan, X., and Wang, X. E. GRIT: Teaching MLLMs to think with images. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025. URL https://openreview.net/forum? id=RW3cHh3HgY.

Fu, Y., Zhu, J., Zhang, L., Zhao, B., Ma, S., Zhang, Y., Wu, Y., and Wu, W. Geolaux: A benchmark for evaluating mllms’ geometry performance on long-step problems requiring auxiliary lines. arXiv preprint arXiv:2508.06226, 2025.

Gao, J., Pi, R., Zhang, J., Ye, J., Zhong, W., Wang, Y., HONG, L., Han, J., Xu, H., Li, Z., and Kong, L. G-LLaVA: Solving geometric problem with multimodal large language model. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/forum? id=px1674Wp3C.

Google. Veo 3 announcement. https://blog. google/innovation-and-ai/products/ generative-media-models-io-2025/, 2025. Accessed January 18, 2026.

Google DeepMind. Gemini 3 pro image (nano banana pro). https://deepmind.google/models/ gemini-image/pro/, November 2025. Accessed January 20, 2026.

Gu, J., Hao, Y., Wang, H. W., Li, L., Shieh, M. Q., Choi, Y., Krishna, R., and Cheng, Y. Thinkmorph: Emergent properties in multimodal interleaved chain-of-thought reasoning. arXiv preprint arXiv:2510.27492, 2025.

Guo, Z., Chen, X., Zhang, R., An, R., Qi, Y., Jiang, D., Li, X., Zhang, M., Li, H., and Heng, P.-A. Are video models ready as zero-shot reasoners? an empirical study with the mme-cof benchmark. arXiv preprint arXiv:2510.26802, 2025.

He, Z., Qu, X., Li, Y., Zhu, T., Huang, S., and Cheng, Y. Diffthinker: Towards generative multimodal reasoning with diffusion models. arXiv preprint arXiv:2512.24165, 2025.

Hessel, J., Holtzman, A., Forbes, M., Le Bras, R., and Choi, Y. Clipscore: A reference-free evaluation metric for image captioning. In Proceedings of the 2021 conference on empirical methods in natural language processing, pp. 7514–7528, 2021.

Hu, E. J., yelong shen, Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., and Chen, W. LoRA: Low-rank adaptation of large language models. In International Conference on Learning Representations, 2022. URL https: //openreview.net/forum?id=nZeVKeeFYf9.

Hu, Y., Shi, W., Fu, X., Roth, D., Ostendorf, M., Zettlemoyer, L., Smith, N. A., and Krishna, R. Visual sketchpad: Sketching as a visual chain of thought for multimodal language models. Advances in Neural Information Processing Systems, 37:139348–139379, 2024.

Huang, K., Guo, J., Li, Z., Ji, X., Ge, J., Li, W., Guo, Y., Cai, T., Yuan, H., Wang, R., Wu, Y., Yin, M., Tang, S., Huang, Y., Jin, C., Chen, X., Zhang, C., and Wang, M. MATHperturb: Benchmarking LLMs’ math reasoning abilities against hard perturbations. In Forty-second International Conference on Machine Learning, 2025. URL https:

//openreview.net/forum?id=OZy70UggXr.

Huang, Z., He, Y., Yu, J., Zhang, F., Si, C., Jiang, Y., Zhang, Y., Wu, T., Jin, Q., Chanpaisit, N., et al. Vbench: Comprehensive benchmark suite for video generative models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 21807–21818, 2024.

Ivanitskiy, M. I., Shah, R., Spies, A. F., R¨auker, T., Valentine, D., Rager, C., Quirke, L., Mathwin, C., Corlouer, G., Behn, C. D., et al. A configurable library for generating and manipulating maze datasets. arXiv preprint arXiv:2309.10498, 2023.

Izadi, A., Banayeeanzade, M., Askari, F., Rahimiakbar, A., Vahedi, M. M., Hasani, H., and Baghshah, M. S. Visual structures help visual reasoning: Addressing the binding problem in LVLMs. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025. URL https://openreview.net/forum? id=T52hZeT7rn.

Ji, A., Kojima, N., Rush, N., Suhr, A., Vong, W. K., Hawkins, R., and Artzi, Y. Abstract visual reasoning with tangram shapes. In Goldberg, Y., Kozareva, Z., and Zhang, Y. (eds.), Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pp. 582–601, Abu Dhabi, United Arab Emirates, December 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.emnlp-main.38. URL https:// aclanthology.org/2022.emnlp-main.38/.

Kahneman, D. Thinking, fast and slow. macmillan, 2011.

Li, A., Wang, C., Fu, D., Yue, K., Cai, Z., Zhu, W. B., Liu, O., Guo, P., Neiswanger, W., Huang, F., et al. Zebracot: A dataset for interleaved vision language reasoning. arXiv preprint arXiv:2507.16746, 2025a.

- Li, B., Sun, X., Liu, J., Wang, Z., Wu, J., Yu, X., Chen, H., Barsoum, E., Chen, M., and Liu, Z. Latent visual reasoning. arXiv preprint arXiv:2509.24251, 2025b.
- Li, C., Zhang, C., Zhou, H., Collier, N., Korhonen, A., and Vuli´c, I. TopViewRS: Vision-language models as top-view spatial reasoners. In Al-Onaizan, Y., Bansal, M., and Chen, Y.-N. (eds.), Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pp. 1786–1807, Miami, Florida, USA, November 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.emnlp-main.

106. URL https://aclanthology.org/2024. emnlp-main.106/.

Li, C., Wu, W., Zhang, H., Li, Q., Gao, Z., Xia, Y., Hern´andez-Orallo, J., Vuli´c, I., and Wei, F. 11plusbench: Demystifying multimodal llm spatial reasoning with cognitive-inspired analysis. arXiv preprint arXiv:2508.20068, 2025c.

Li, C., Wu, W., Zhang, H., Xia, Y., Mao, S., Dong, L., Vuli´c, I., and Wei, F. Imagine while reasoning in space: Multimodal visualization-of-thought. In Fortysecond International Conference on Machine Learning, 2025d. URL https://openreview.net/forum? id=6vk6Xg24ZC.

Liu, F., Wang, H., Cai, Y., Zhang, K., Zhan, X., and Duan, Y. Video-t1: Test-time scaling for video generation. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pp. 18671–18681, October 2025.

Muennighoff, N., Yang, Z., Shi, W., Li, X. L., Fei-Fei, L., Hajishirzi, H., Zettlemoyer, L., Liang, P., Cand`es, E., and Hashimoto, T. s1: Simple test-time scaling, 2025. URL https://arxiv.org/abs/2501.19393.

OpenAI. Learning to reason with llms. https://openai.com/index/ learning-to-reason-with-llms/, September 2024.

OpenAI. Gpt-5.1: A smarter, more conversational chatgpt. https://openai.com/index/gpt-5-1/, November 2025a. Accessed January 20, 2026.

OpenAI. Introducing gpt-5.2. https://openai. com/index/introducing-gpt-5-2/, December 2025b. Accessed January 20, 2026.

Su, A., Wang, H., Ren, W., Lin, F., and Chen, W. Pixel reasoner: Incentivizing pixel space reasoning via curiositydriven reinforcement learning. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025. URL https://openreview.net/forum? id=VeZkY3JjWV.

Team, G. R., Devin, C., Du, Y., Dwibedi, D., Gao, R., Jindal, A., Kipf, T., Kirmani, S., Liu, F., Majumdar, A., et al. Evaluating gemini robotics policies in a veo world simulator. arXiv preprint arXiv:2512.10675, 2025.

Tong, C., Chang, M., Zhang, S., Wang, Y., Liang, C., Zhao, Z., An, R., Zeng, B., Shi, Y., Dai, Y., et al. Cof-t2i: Video models as pure visual reasoners for text-to-image generation. arXiv preprint arXiv:2601.10061, 2026.

Tong, J., Gu, J., Lou, Y., Fan, L., Zou, Y., Wu, Y., Ye, J., and Li, R. Sketch-in-latents: Eliciting unified reasoning in mllms. arXiv preprint arXiv:2512.16584, 2025a.

Tong, J., Mou, Y., Li, H., Li, M., Yang, Y., Zhang, M., Chen, Q., Liang, T., Hu, X., Zheng, Y., et al. Thinking with video: Video generation as a promising multimodal reasoning paradigm. arXiv preprint arXiv:2511.04570, 2025b.

Wan, T., Wang, A., Ai, B., Wen, B., Mao, C., Xie, C.-W., Chen, D., Yu, F., Zhao, H., Yang, J., et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025.

Wang, Z., Jiang, D., Li, L., Dang, S., Li, C., Yang, H., Dai, G., Wang, M., and Wang, J. Deforming videos to masks: Flow matching for referring video segmentation. arXiv preprint arXiv:2510.06139, 2025.

Wei, J., Bosma, M., Zhao, V., Guu, K., Yu, A. W., Lester, B., Du, N., Dai, A. M., and Le, Q. V. Finetuned language models are zero-shot learners. In International Conference on Learning Representations, 2022a. URL https: //openreview.net/forum?id=gEZrGCozdqR.

Wei, J., Wang, X., Schuurmans, D., Bosma, M., Xia, F., Chi, E., Le, Q. V., Zhou, D., et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824–24837, 2022b.

Wiedemer, T., Li, Y., Vicol, P., Gu, S. S., Matarese, N., Swersky, K., Kim, B., Jaini, P., and Geirhos, R. Video models are zero-shot learners and reasoners. arXiv preprint arXiv:2509.20328, 2025.

- Wu, B., Zou, C., Li, C., Huang, D., Yang, F., Tan, H., Peng, J., Wu, J., Xiong, J., Jiang, J., et al. Hunyuanvideo 1.5 technical report. arXiv preprint arXiv:2511.18870, 2025a.
- Wu, C., Li, J., Zhou, J., Lin, J., Gao, K., Yan, K., ming Yin, S., Bai, S., Xu, X., Chen, Y., Chen, Y., Tang, Z., Zhang, Z., Wang, Z., Yang, A., Yu, B., Cheng, C., Liu, D., Li,
- D., Zhang, H., Meng, H., Wei, H., Ni, J., Chen, K., Cao, K., Peng, L., Qu, L., Wu, M., Wang, P., Yu, S., Wen, T., Feng, W., Xu, X., Wang, Y., Zhang, Y., Zhu, Y., Wu, Y.,

Cai, Y., and Liu, Z. Qwen-image technical report, 2025b. URL https://arxiv.org/abs/2508.02324.

Wu, J., Huang, T., He, C., and Long, M. Miniveo3-reasoner: Thinking with videos from open-source priors. https: //github.com/thuml/MiniVeo3-Reasoner, 2025c.

Xu, Y., Li, C., Zhou, H., Wan, X., Zhang, C., Korhonen, A., and Vuli´c, I. Visual planning: Let’s think only with images. arXiv preprint arXiv:2505.11409, 2025.

Yang, A., Li, A., Yang, B., Zhang, B., Hui, B., Zheng, B., Yu, B., Gao, C., Huang, C., Lv, C., et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025a.

Yang, C., Wan, H., Peng, Y., Cheng, X., Yu, Z., Zhang, J., Yu, J., Yu, X., Zheng, X., Zhou, D., et al. Reasoning via video: The first evaluation of video models’ reasoning abilities through maze-solving tasks. arXiv preprint arXiv:2511.15065, 2025b.

Yang, R., Chen, H., Zhang, J., Zhao, M., Qian, C., Wang, K., Wang, Q., Koripella, T. V., Movahedi, M., Li, M., Ji, H., Zhang, H., and Zhang, T. Embodiedbench: Comprehensive benchmarking multi-modal large language models for vision-driven embodied agents. In Fortysecond International Conference on Machine Learning, 2025c. URL https://openreview.net/forum? id=DgGF2LEBPS.

Zhang, H., Li, C., Wu, W., Mao, S., Zhang, Y., Tian, H., Vuli´c, I., Zhang, Z., Wang, L., Tan, T., et al. Scaling and beyond: Advancing spatial reasoning in mllms requires new recipes. arXiv preprint arXiv:2504.15037, 2025a.

Zhang, H., Wu, W., Li, C., Shang, N., Xia, Y., Huang, Y., Zhang, Y., Dong, L., Zhang, Z., Wang, L., et al. Latent sketchpad: Sketching visual thoughts to elicit multimodal reasoning in mllms. arXiv preprint arXiv:2510.24514, 2025b.

Zheng, Z., Yang, M., Hong, J., Zhao, C., Xu, G., Yang, L., Shen, C., and Yu, X. Deepeyes: Incentivizing” thinking with images” via reinforcement learning. arXiv preprint arXiv:2505.14362, 2025.

### A. Experimental Setups

#### A.1. Dataset

MAZENAVIGATION We evaluate models across a suite of grid-world maze navigation environments (Ivanitskiy et al., 2023), with dimensions ranging from 3 × 3 to 8 × 8 and optimal planning lengths spanning 2–18 steps. Each configuration features a unique spatial layout, with the agent’s initial state v0 sampled from reachable grid locations relative to the goal. We follow the practice in Xu et al. (2025) where the agent location is randomly initialized, and the solution is obtained from the built-in searching algorithm from Ivanitskiy et al. (2023).

For the training set, we generate 1,000 unique instances for maze sizes from 3 × 3 to 6 × 6, and maximum navigation path length ≤ 12. To ensure visual diversity and complexity, we use 40 unique visual icons instead of the unique one icons, as shown in Figure 4. It is worth noting that we do not explicitly replicate maze layouts with different visual icons; consequently, the same environment does not appear with multiple agent icon variants. To assess the model’s robustness and systematic generalization, we compile the held-out evaluation set into three difficulty tiers based on structural and temporal complexity (see Table 4):

- • In-Distribution: Test cases share the same distribution of grid sizes and planning lengths as the training set, featuring novel maze layouts and randomized start/goal configurations.
- • OOD Maze Sizes (Spatial OOD): Environments scale to 7 × 7 and 8 × 8 dimensions while maintaining distribution of training-level navigation path steps, evaluating adaptability to unseen spatial scales.
- • OOD Path Length (Temporal OOD): We increase the path length as the indication of reasoning complexity to 13–18 steps across in-distribution grid sizes (5 × 5, 6 × 6), specifically challenging long-range dependency modeling and compositional reasoning.
- • OOD Both (Temporal & Spatial OOD): We increase both maze sizes and path length to 13–18 steps across grid sizes from 7 × 7 to 8 × 8.

In parallel, we construct a visually OOD test set featuring icons that are unseen during training (right panel of Figure 4). The instance-wise maze layouts are identical across icon sets, enabling a controlled evaluation of the model’s ability to generalize to unseen visual appearances.

Icons for Training Unseen Icons

[Figure 22]

[Figure 23]

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

Figure 4. Agent Icons for MAZENAVIGATION during training and visual OOD evaluation.

TANGRAMPUZZLE We introduce a Tangram-based puzzle-solving task derived from the Kilogram dataset (Ji et al., 2022). To construct our dataset, we sample all black-background silhouettes from the training set of Kilogram and remove the underlying grid structures, for a total of 692 data items. For each silhouette we provide seven disjoint, color-coded geometric pieces, and every silhouette has a unique ground-truth layout. Based on different degrees of geometric transformation and whether to provide visual context, we introduce three variants of TANGRAMPUZZLE: Fade-In, Rotation, and Translation, as shown by Figure 5.

Table 4. Statistics of MAZENAVIGATION across types

Type Category Grid Sizes Path Length Numbers Train set - 3 × 3 to 6 × 6 2 – 12 steps 4,000 (4 × 1,000) Test set In-Distribution 3 × 3 to 6 × 6 2 – 12 steps 1,000 (4 × 250)

Spatial OOD 7 × 7 to 8 × 8 2 – 12 steps 500 (2 × 250) Temporal OOD 5 × 5 to 6 × 6 13 – 18 steps 500 (2 × 250) Spatiotemporal OOD 7 × 7 to 8 × 8 13 – 18 steps 500 (2 × 250)

- • Fade-In: Pieces are initialized in their canonical orientations. The model’s objective is limited to localizing target centroids within the silhouette.
- • Rotation: Pieces are initialized with random orientations, listed on the left sidebar. The model must perform sequential SO(2) rotations followed by spatial translations to reconstruct the target silhouette.
- • Translation: Pieces are initialized in their canonical orientations in the left sidebar. The model’s objective is limited to localizing target centroids within the silhouette.

For evaluation, we use the held-out split of the Kilogram dataset (125 data items) to ensure zero layout overlap of silhouette patterns between training and test distributions.

#### A.2. Models and Hyper-Parameters

To investigate the impact of modalities on planning performance, we perform supervised fine-tuning (SFT) across three distinct modalities: text, image, and video.

Planning in text. In this formulation, the model leverages natural language to represent discrete action sequences in MAZENAVIGATION and use coordinates and rotation degrees in JSON format for TANGRAMPUZZLE. Formally, given a visual state v ∈ V and a task-oriented textual prompt p, the model is trained to generate a textual plan t = (t1,...,tL), where each token ti belongs to the language vocabulary Vtext. The input is constructed by concatenating the visual tokens with the prompt tokens. Following standard protocols for autoregressive supervised fine-tuning (Wei et al., 2022a), we optimize model parameters θ by minimizing the negative log-likelihood of the target action sequence:

LSFT(θ) = −E(v,p,t)∼D

L

log πθ ti | t<i,v,p (1)

i=1

where D denotes the training distribution. This formulation treats planning as a conditional sequence-generation task, grounding the language-based action space in the visual context.

For implementation, we evaluate proprietary models GPT-5.1 (OpenAI, 2025a) and GPT-5.2 (OpenAI, 2025b)1 on MAZENAVIGATION. For open-source baselines, we fine-tune Qwen 2.5 VL 7B (Bai et al., 2025b) with full parameter updates on both tasks. For TANGRAMPUZZLE, the Qwen model is trained to output 2D center coordinates and floating-point rotation degrees in a strict JSON format (prompt templates in Appendix C).

Planning in image. For MAZENAVIGATION, we adopt the Visual Planning framework (Xu et al., 2025), where the solution is generated as a sequence of discrete images (one frame per step). To ensure a rigorous baseline, we strictly replicate the architecture, reward design, and single-icon (blue star) training distribution from Xu et al. (2025). We restrict this comparison to MAZENAVIGATION to avoid implementation biases that might arise from adapting their specialized reward model to TANGRAMPUZZLE.

For TANGRAMPUZZLE, we treat planning as a “direct prediction” problem using Image Editing. We first evaluate the proprietary model (Nano-Banana, formally, Gemini 3 Pro Image (Google DeepMind, 2025)). Due to the financial constraint, we only evaluate the first 51 instances in TANGRAMPUZZLE test set with Nano Banana. For open-sourced models, we also

1We utilize GPT-5.1 (2025-11-13) and GPT-5.2 (2025-12-11) via the Azure API.

Fade-In

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

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

Rotation

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

Translation

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

Figure 5. Illustration of different variants for TANGRAMPUZZLE.

experiment with Qwen Image Edit 2509 (Wu et al., 2025b) (20B parameters), a state-of-the-art open-source image editing model. We fine-tune the model using LoRA (Hu et al., 2022) with default hyperparameters. Input resolutions are set to 256 × 256 for the Fade-In setting and 256 × 512 for Rotation/Translation settings to accommodate the sidebar workspace.

Planning in video. We use Wan 2.2 TI2V 5B as the main model backbone for experiments, which is a text-to-video diffusion model conditional on textual instructions. It’s pretrained on large-scale video data, each with sequences of 81 frames. For experiments, we fine-tune the model with LoRA (Hu et al., 2022) for 20 epochs for both MAZENAVIGATION and TANGRAMPUZZLE. Specifically, in MAZENAVIGATION, we train the model on 81-frame sequences, aligned with the pre-trained video length, with a fixed resolution of 480 × 832. In TANGRAMPUZZLE, preliminary experiments revealed that compressing the complex rotation and translation of Tangram pieces into 81 frames resulted in rapid motion blurring, which degraded geometric consistency. Apart from adhering to the pre-trained video length in the Fade-In setting, to mitigate this, we extended the sequence length to 201 frames for the Rotation setting. For Translation, we employed random frame lengths (sampled from 61 to 81 with a stride of 4) as data augmentation to improve robustness.

Table 5. Hyperparameters for supervised fine-tuning across different modalities.

#### Hyperparameter Text Image Video

Epochs 20 20 20 Learning rate 1 × 10−5 1 × 10−4 1 × 10−4 Train batch size 16 1 1 Gradient accumulation 1 1 1 GPUs 2 1 1

LoRA rank 128 32 32 LoRA modules q proj,k proj,v proj

to q, to k, to v, to out.0 add q proj, add k proj add v proj, to add out img mlp.net.2, img mod.1 txt mlp.net.2, txt mod.1

q, k, v, o ffn.0, ffn.2

o proj, gate proj, down proj, up proj

All diffusion models are trained using the DiffSynth-Studio framework2. Detailed hyperparameters are provided in Table 5.

#### A.3. Evaluator

MAZENAVIGATION To evaluate the reasoning accuracy across diverse agent appearances and stochastic generation speeds, we designed a pipeline prioritizing visual robustness and temporal invariance based on the implementation from Wu et al. (2025c). First, to handle varying agent designs (different icons) without training specific detectors, we adopt a motion-centric extraction approach. By modeling the static maze environment as a reference, we isolate the agent solely based on its dynamics via background subtraction. To further mitigate generative artifacts, such as flickering or hallucinated objects, we impose spatiotemporal continuity constraints, restricting the tracker to physically plausible local neighborhoods. Second, addressing the variable pacing inherent in diffusion models, we reject frame-level comparisons (pt vs. pgtt ) as they are sensitive to speed mismatches. Instead, we implement a speed-invariant alignment protocol. By resampling trajectories based on cumulative geometric distance rather than time indices, we decouple spatial path fidelity from temporal dynamics. This ensures the metric rigorously evaluates where the agent went, regardless of how fast it travelled relative to the ground truth. Based on the implementation above, we adopt the evaluation metric (EM for Exact Match and PR for Progress Rate) from Xu et al. (2025).

TANGRAMPUZZLE To rigorously assess the generative model’s reasoning capabilities without bias, we developed a deterministic, rule-based visual evaluator. Unlike neural-based evaluators (e.g., CLIPScore (Hessel et al., 2021)), which correlate poorly with precise geometric constraints, our pipeline operates on pixel-level segmentation and geometric primitives. The evaluation pipeline processes the generated video V = {v0,...,vT} and extracts metrics based on color segmentation, contour approximation, and temporal consistency.

We define the set of N = 7 tangram pieces based on a predefined palette of unique colors C = {c1,...,cN}, where for Fade-In, the colors are fixed in the whole dataset, for Translation and Rotation, the colors are randomly assigned to each tangram pieces as they are listed on the left side of the image as visual context.

We first establish ground-truth physical geometric properties for each piece i. For Fade-In, the geometric properties of the tangram pieces could be obtained from the golden target layouts; for Translation and Rotation, the geometric properties are obtained from the left side of the first frame (input frame). The properties include:

- • Initial Area (A(0)i ): The pixel count of the color mask in the source region.
- • Initial Shape Class (Si(0)): The geometric topology (triangle, square, parallelogram) identified via contour approximation.
- • Target Silhouette (Mtgt): The binary mask of the black target shape in the target region. Specifically, to identify shapes under potential generation artifacts (e.g., anti-aliasing or slight warping), we use

2https://github.com/modelscope/DiffSynth-Studio

cv2.approxPolyDP to approximate the contour of each segmented piece into vertices. We classify the shape Si based on the number of vertices, internal angles, and aspect ratios:

- • Triangle: 3 vertices.
- • Square: 4 vertices, aspect ratio ≈ 1.0, and all internal angles θ ∈ [85◦,95◦].
- • Parallelogram: 4 vertices, opposite sides equal in length, and opposite internal angles equal (with a tolerance of ±15◦). Based on the implementation above, we define the metrics in Section 3.2 as follows.

Goal Completion This metric evaluates the final state of the puzzle in the last frame vT. A trial is considered successful if the average piece-wise completion score is 1.0. For each piece i, the completion score ui ∈ [0,1] is calculated based on following constraints: (1) Area Consistency: The piece must be present in the target region with an area A(iT) within a tolerance window of the initial area to account for rotation-induced aliasing: 0.6 ≤ A

(T ) i

≤ 1.4. (2) Shape Preservation:

A(0)i

The classified shape class must match the ground truth: Si(T) = Si(0). (3) Within the silhouette: The shape must be within the target silhouette. Only if the piece satisfies the three constraints above would it be considered correct.

Boundary Adherence (Mask IoU) This metric quantifies how well the generated pieces fit inside the target silhouette without hallucinating pixels outside the lines. We compute the union of all generated color masks in the final frame,

Mgen = Ni=1 Mi(T). The score is defined as the pixel-wise precision relative to the target silhouette mask Mtgt:

SIoU = |Mgen ∩ Mtgt| |Mgen ∪ Mtgt|

- A score of 1.0 implies the generated puzzle perfectly fills the silhouette with zero overflow and zero gaps.

Indicative Visual Consistency (Piece Integrity) To measure whether the model respects “object permanence” throughout the video (rather than pieces vanishing or flickering), we compute a temporal integrity score. We sample frames evenly across the video sequence. For each sampled frame vt, we calculate the ratio of total visible area for each color (summed across both source and target regions) relative to the initial area. Let ri,t = A(it)/A(0)i . We define a binary integrity flag δi,t = I(0.6 ≤ ri,t ≤ 1.4). The final integrity score is calculated by comparing the sequence of integrity flags against a reference ground-truth video (if available) or the ideal static assumption, quantifying the fraction of frames where all pieces maintain physical consistency. To be noted that this metric only serves as an indicative metric for analysis. Therefore, it is not included in the main result table.

- B. Results

Fine-Grained Results for MAZENAVIGATION. Supplement to Table 2, we illustrate the fine-grained in-distribution performance gap between general proprietary models and specialized finetuned MLLMs in MAZENAVIGATION tasks in Table 6. A consistent inverse correlation is observed between maze complexity and model performance; as grid size scale from 3×3 to 6×6, all models exhibit a systematic degradation in EM and PR scores, highlighting the inherent difficulty of larger maze planning. Notably, the proprietary GPT-5 series struggles to surpass a 20% success rate. In contrast, Wan2.2 TI2V 5B model achieves near-saturated performance (up to 99.98% PR), demonstrating that internalizing spatial dynamics through video-based output modalities provides a superior inductive bias for such visual-first navigation task.

Qualitative Analysis for MAZENAVIGATION. Figure 11 and Figure 12 show the correct and wrong generations from video generation model in MAZENAVIGATION. We observe that Wan 2.2 TI2V 5B demonstrates robust object permanence throughout the generated video with high accuracy for both seen icons and unseen icons across ID and OOD settings. This underscores a robust zero-shot generalization capability that transcends simple pixel-level memorization, positioning multimodal generative priors as a more viable backbone for complex visual-first reasoning tasks than traditional MLLMs. To quantitatively analyze the failure cases, we categorize observed failures into distinct taxonomies. For MAZENAVIGATION, we identify four primary failure modes: 1) boundary violation (crossing the walls), 2) structural distortion (unstable generation of visual icons), 3) kinematic inconsistency (agent disappearance or teleportation), and 4) wrong movement actions. Qualitatively, we observe that similar visual semantics (e.g., similar colors) between agent icon and destination icon are more prone to error such as kinematic inconsistency, or even a minor case where the model confuses the agent icon with the goal icon, indicating the importance of visual semantics.

Table 6. Evaluation results of various models across different maze sizes and path lengths. represents texts, represents images, and represents videos. ■ represents visual reasoning systems.

3x3 4x4 5x5 6x6 EM ↑ PR ↑ EM ↑ PR ↑ EM ↑ PR ↑ EM ↑ PR ↑ Proprietary Models

Model Input Output

- GPT-5.1 + 15.6 16.0 11.6 11.6 8.40 8.40 6.80 6.80
- GPT-5.2 + 18.4 18.4 13.2 13.2 10.0 10.0 8.40 8.40 Open-Sourced Models Qwen2.5-VL-7B + 88.8 91.5 62.0 69.6 45.2 57.6 23.6 39.0

- - w CoT 83.6 85.8 53.6 61.8 40.8 51.9 24.0 36.4 Qwen3-VL-8B + 89.2 91.5 69.6 76.7 44.8 60.0 29.6 46.2
- - w CoT 94.4 94.6 79.6 83.5 64.0 70.8 50.0 60.4 VPRL-7B * 94.0 96.0 72.0 76.0 66.0 74.4 62.0 68.0 Wan2.2-TI2V-5B + 96.0 99.8 98.0 99.4 98.0 99.9 92.0 94.8

- - Unseen Visual Icons 94.0 98.6 98.0 99.4 96.0 99.9 94.0 94.8

Qualitative Analysis for TANGRAMPUZZLE. Figure 13 shows the visualized predictions from different systems in different settings. For Qwen VL with textual reasoning, the model fails to output precise continuous coordinates, yielding zero strict goal completion, although being specifically tuned. This indicates that the text modality struggles to describe continuous fine-grained rotation and translation, although it doesn’t need to maintain the geometric consistency to manipulate the pieces with coordinates. For image editing model, as discussed in Section 5.3, we show that after being supervised fine-tuned on the corresponding tasks, it can achieve much better completion results in TANGRAMPUZZLE compared to proprietary models which struggles to maintain the shape of tangram pieces, as shown in Figure 13. Fine-tuned video generation model excels in the Translation setting but struggles to maintain the geometric properties of tangram pieces in Fade-In and Rotation. We identify the primary failure modes by geometric and visual fidelity as follows: chromatic distortion (pieces change color and shape during the process, which is most evident for visual generative models), centroid displacement (the piece is placed away from the target location, which is most evident for textual reasoning models), and angular deviation (incorrect rotational orientation). These failure cases are qualitatively illustrated in Figures 13.

Our error analysis reveals distinct bottlenecks across both domains. Although video generation models can adapt to physical environment rules (e.g., not crossing walls) in MAZENAVIGATION, they struggle to maintain structural fidelity of visual semantics. This issue manifests in both MAZENAVIGATION, where changes in visual semantics often lead to errors, and TANGRAMPUZZLE, where geometric distortions accumulate over time in the video. These findings suggest that, despite seemingly strong agent-level reasoning, sustaining long-term consistency of visual semantics remains a fundamental challenge for visual generative models when used as visual reasoners. This limitation has been largely obscured by prior work that focuses primarily on local visual changes, and we therefore call for further research in this direction.

- C. Prompt Prompt of Maze Task (Text Version) Task : Maze Shortest Path Planning You are given an image of a maze environment. In this environment :

- - An icon marks the starting position of the agent.
- - A red point marks the goal.
- - The agent can move in one of four cardinal directions only : ”up” , ”down” , ”left” , or ”right”. Each move shifts the agent by exactly one cell in that direction. Diagonal movement is not permitted.
- - The black maze walls are impassable. The agent cannot pass through any wall segment. Your task is to analyse the image and produce the shortest valid sequence of actions that moves the agent from its starting position to the goal without crossing any wall. Provide your final answer in the end enclosed between <ANSWER> and </ANSWER>, for example: <ANSWER> right up up </ANSWER>.

105

100

95

90

- 7x7

EM PR

61 81 101 121 141

Frames

20

30

40

50

60

70

80

90

5x5LongerPath

EM PR

61 81 101 121 141

Frames

20

30

40

50

60

70

7x7LongerPath

EM PR

61 81 101 121 141

Frames

55

60

65

70

75

80

85

90

- 8x8

85

80

75

70

61 81 101 121 141

Frames

50

EM PR

EM PR

EM PR

80

45

70

40

6x6LongerPath

8x8LongerPath

35

60

30

50

25

40

20

15

30

10

61 81 101 121 141

61 81 101 121 141

Frames

Frames

Figure 6. Fine-Grained Visual Test-Time Scaling for MAZENAVIGATION using unseen icon with more inference frame budget.

##### Prompt of Maze Task (Video Version)

Create a 2D animation based on the provided image of a maze. The custom character slides smoothly along the white path, stopping perfectly on the red circle destination. The character never slides or crosses into the black segments of the maze. The camera is a static, top-down view showing the entire maze. Maze:

- * The maze paths are white, the walls are black.
- * The character starts from its initial position.
- * The character slides smoothly along the white path.
- * The character never slides or crosses into the black segments of the maze.
- * The character stops perfectly on the red circle. Scene:
- * No change in scene composition.
- * No change in the layout of the maze.
- * The character travels along the path without speeding up or slowing down. Camera:
- * Static camera.
- * No zoom.
- * No pan.
- * No glitches, noise, or artifacts.

Prompt of Tangram Task (Text Version)

Analyze the provided Tangram puzzle image in which the silhouette can be filled by the seven standard Tangram pieces. Puzzle:

- * The silhouette of the puzzle is represented as black area in the white background.
- * The Tangram pieces include: 2 big triangles, 1 medium triangle, 2 small triangles, 1 square, and 1 parallelogram.
- * The shapes of the pieces can not be altered. Colors:
- * Every piece has a distinct, unique color, as shown on the left side. Question: Identify the final center coordinates for each piece after it is positioned within the silhouette.

100

95

90

85

- 7x7

EM PR

5 7 9 11

Factor

30

40

50

60

70

80

5x5LongerPath

EM PR

5 7 9 11

Factor

30

40

50

60

70

7x7LongerPath

EM PR

5 7 9 11

Frames

50

55

60

65

70

75

80

85

- 8x8

80

75

70

65

60

5 7 9 11

Factor

90

55

EM PR

EM PR

EM PR

50

80

45

6x6LongerPath

8x8LongerPath

40

70

35

30

60

25

20

50

15

40

10

5 7 9 11

5 7 9 11

Frames

Frames

- Figure 7. Fine-Grained Visual Test-Time Scaling for MAZENAVIGATION using unseen icon with more inference budget determined by the control variable scaling factor κ.

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

Similar to "Self-Correction"

- Figure 8. Example of trajectory similar to “self-correction” in MAZENAVIGATION when provided with more inference frame budget.

Prompt of Tangram Task (Image Editing Version) Fade-In

Solve the tangram puzzle by arranging the pieces to exactly match the black silhouette. Do not change the color or shape of any tangram piece, and do not modify the overall silhouette. All pieces must be placed entirely within the silhouette, without overlapping or extending beyond its boundaries.

The colors of the tangram pieces are as follows:

- * Large triangles: blue and orange
- * Medium triangle: green
- * Small triangles: purple and yellow
- * Square: gray
- * Parallelogram: red Rotation & Translation

Use the tangram pieces shown on the left to exactly fill the black tangram silhouette on the right. Do not change the color, size, or shape of any tangram piece, and do not alter the overall silhouette of the puzzle. All pieces

must be placed entirely within the silhouette, without overlapping each other or extending beyond its boundaries.

Prompt of Tangram Task (Video Version) Fade-In

Create a 2D animation showing the step-by-step assembly of a Tangram puzzle. Puzzle:

- * The silhouette of the puzzle is represented as black area in the white background.
- * The Tangram pieces include: 2 big triangles, 1 medium triangle, 2 small triangles, 1 square, and 1 parallelogram.
- * The pieces appear one by one, fading in from transparent to their specific colors to fill the silhouette. Colors:
- * Big triangles: blue and orange
- * Small triangles: purple and yellow
- * Medium triangle: green
- * Square: grey
- * Parallelogram: red Scene:
- * No change in scene composition.
- * No change in the silhouette of the puzzle. Camera:
- * Static camera.
- * No zoom.
- * No pan.
- * No glitches, noise, or artifacts.

Rotation

Create a 2D animation showing the step-by-step assembly of a Tangram puzzle. Puzzle:

- * The silhouette of the puzzle is represented as black area in the white background.
- * The Tangram pieces include: 2 big triangles, 1 medium triangle, 2 small triangles, 1 square, and 1 parallelogram.
- * The shapes of the pieces can not be altered.
- * Sequential accumulation constraint: Pieces move and orient individually one by one. Upon placement, every piece is permanently locked in place with its unique color and orientation. Continue until the silhouette is full. Colors:
- * Every piece has a distinct, unique color, as shown on the left side. Scene:
- * No change in scene composition.
- * No change in the silhouette of the puzzle.
- * No change in the designated colors and shapes of the pieces. Camera:
- * Static camera.
- * No zoom.
- * No pan.
- * No glitches, noise, or artifacts.

Translation Create a 2D animation showing the step-by-step assembly of a Tangram puzzle.

Puzzle:

- * The silhouette of the puzzle is represented as black area in the white background.
- * The Tangram pieces include: 2 big triangles, 1 medium triangle, 2 small triangles, 1 square, and 1 parallelogram.
- * The shapes of the pieces can not be altered.
- * Sequential accumulation constraint: Pieces move individually one by one. Upon placement, every piece is permanently locked in place with its unique color and orientation. Continue until the silhouette is full. Colors:
- * Every piece has a distinct, unique color, as shown on the left side. Scene:
- * No change in scene composition.
- * No change in the silhouette of the puzzle.
- * No change in the designated colors and shapes of the pieces. Camera:
- * Static camera.
- * No zoom.
- * No pan.
- * No glitches, noise, or artifacts.

Zero-Shot Inference on Irregular Mazes

Success Case

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

[Figure 178]

[Figure 179]

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

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

[Figure 199]

[Figure 200]

[Figure 201]

Failure Case

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

- Figure 9. Visualizations of zero-shot inference on irregular maze with Wan 2.2 TI2V 5B trained on regular mazes. We observe that although not included in the training data, trained video generation model can adapt a certain level of planning capabilities to such irregular mazes with different background and meanwhile keeping the constraint (no change in the background, follow the path, do not cross the wall), and surprisingly can move diagonally.

OOD

100

90

80

Score

70

60

50

40

Strict GC

Progress GC

30

61 81 89 101 121

Frames

- Figure 10. We don’t observe similar Visual Test-Time Scaling trend in TANGRAMPUZZLE, possibly due to the constraint of geometric consistency throughout the generated video that hinders the performance. However, we still observe that by providing more inference frames (81-121 frames) compared to training setting (61-81 frames) would not severely harm the performance.

Seen Icons During Training

In-Distribution Maze Size & Path Length

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

OOD Maze Size

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

OOD Path Length

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

OOD Both

[Figure 248]

[Figure 249]

[Figure 250]

[Figure 251]

[Figure 252]

[Figure 253]

[Figure 254]

[Figure 255]

[Figure 256]

[Figure 257]

Unseen Icons

In-Distribution Maze Sizes & Path Length

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

[Figure 270]

[Figure 271]

[Figure 272]

[Figure 273]

[Figure 274]

[Figure 275]

[Figure 276]

[Figure 277]

OOD Maze Size

[Figure 278]

[Figure 279]

[Figure 280]

[Figure 281]

[Figure 282]

[Figure 283]

[Figure 284]

[Figure 285]

[Figure 286]

[Figure 287]

OOD Path Length

[Figure 288]

[Figure 289]

[Figure 290]

[Figure 291]

[Figure 292]

[Figure 293]

[Figure 294]

[Figure 295]

[Figure 296]

[Figure 297]

OOD Both

[Figure 298]

[Figure 299]

[Figure 300]

[Figure 301]

[Figure 302]

[Figure 303]

[Figure 304]

[Figure 305]

[Figure 306]

[Figure 307]

[Figure 308]

[Figure 309]

[Figure 310]

[Figure 311]

[Figure 312]

[Figure 313]

[Figure 314]

[Figure 315]

[Figure 316]

[Figure 317]

Figure 11. Correct examples of MAZENAVIGATION generated by fine-tuned Wan 2.2 TI2V 5B.

Seen Icons During Training

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

Cross the wall

[Figure 328]

[Figure 329]

[Figure 330]

[Figure 331]

[Figure 332]

[Figure 333]

[Figure 334]

[Figure 335]

[Figure 336]

[Figure 337]

Disappearance & Teleportation

[Figure 338]

[Figure 339]

[Figure 340]

[Figure 341]

[Figure 342]

[Figure 343]

[Figure 344]

[Figure 345]

[Figure 346]

[Figure 347]

Distortion

Cross the wall

Unseen Icons

[Figure 348]

[Figure 349]

[Figure 350]

[Figure 351]

[Figure 352]

[Figure 353]

[Figure 354]

[Figure 355]

[Figure 356]

[Figure 357]

Disappearance & Teleportation

[Figure 358]

[Figure 359]

[Figure 360]

[Figure 361]

[Figure 362]

[Figure 363]

[Figure 364]

[Figure 365]

[Figure 366]

[Figure 367]

Wrong moving agent Disappearance & Teleportation

[Figure 368]

[Figure 369]

[Figure 370]

[Figure 371]

[Figure 372]

[Figure 373]

[Figure 374]

[Figure 375]

[Figure 376]

[Figure 377]

Cross the wall

Figure 12. Wrong examples of MAZENAVIGATION generated by fine-tuned Wan 2.2 TI2V 5B.

Fade-In

[Figure 378]

[Figure 379]

Qwen Image Edit 2509

Wan 2.2 TI2V 5B

[Figure 380]

[Figure 381]

[Figure 382]

[Figure 383]

[Figure 384]

[Figure 385]

[Figure 386]

[Figure 387]

[Figure 388]

|[Figure 389]|
|---|

|[Figure 390]|
|---|

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

Rotation

[Figure 402]

[Figure 403]

[Figure 404]

[Figure 405]

Qwen-VL

Nano Banana Qwen Image Edit 2509

v

Wan 2.2 TI2V 5B

[Figure 406]

[Figure 407]

[Figure 408]

[Figure 409]

[Figure 410]

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

[Figure 426]

[Figure 427]

[Figure 428]

[Figure 429]

[Figure 430]

[Figure 431]

[Figure 432]

[Figure 433]

[Figure 434]

[Figure 435]

[Figure 436]

[Figure 437]

[Figure 438]

[Figure 439]

[Figure 440]

Translation

[Figure 441]

[Figure 442]

[Figure 443]

| |
|---|

Qwen-VL Nano Banana Qwen Image Edit 2509

Wan 2.2 TI2V 5B

[Figure 444]

[Figure 445]

[Figure 446]

[Figure 447]

[Figure 448]

[Figure 449]

[Figure 450]

[Figure 451]

[Figure 452]

[Figure 453]

[Figure 454]

[Figure 455]

[Figure 456]

[Figure 457]

[Figure 458]

[Figure 459]

[Figure 460]

[Figure 461]

[Figure 462]

[Figure 463]

[Figure 464]

[Figure 465]

[Figure 466]

[Figure 467]

[Figure 468]

[Figure 469]

[Figure 470]

[Figure 471]

[Figure 472]

[Figure 473]

[Figure 474]

[Figure 475]

Figure 13. Examples of TANGRAMPUZZLE generated by different systems. Red bounding boxes indicates wrong predictions, and green bounding box indicates distortion throughout the process.

