# arXiv:2603.13033v1[cs.CV]13Mar2026

## ESPIRE: A DIAGNOSTIC BENCHMARK FOR EMBODIED SPATIAL REASONING OF VISION-LANGUAGE MODELS

Yanpeng Zhao† Wentao Ding† Hongtao Li† Baoxiong Jia Zilong Zheng State Key Laboratory of General Artificial Intelligence, BIGAI

https://spatigen.github.io/espire.io/ https://github.com/spatigen/espire

ABSTRACT

A recent trend in vision-language models (VLMs) has been to enhance their spatial cognition for embodied domains. Despite progress, existing evaluations have been limited both in paradigm and in coverage, hindering rapid, iterative model development. To address these limitations, we propose ESPIRE, a diagnostic benchmark for embodied spatial reasoning. ESPIRE offers a simulated world that physically grounds VLMs and evaluates them on spatial-reasoning-centric robotic tasks, thus narrowing the gap between evaluation and real-world deployment. To adapt VLMs to robotic tasks, we decompose each task into localization and execution, and frame both as generative problems, in stark contrast to predominant discriminative evaluations (e.g., via visual-question answering) that rely on distractors and discard execution. This decomposition further enables a fine-grained analysis beyond passive spatial reasoning toward reasoning to act. We systematically design ESPIRE both at the instruction level and at the environment level, ensuring broad coverage of spatial reasoning scenarios. We use ESPIRE to diagnose a range of frontier VLMs and provide in-depth analysis of their spatial reasoning behaviors.

1 INTRODUCTION

Spatial cognition goes beyond perception; it enables reasoning and interaction with the 3D physical world, forming the foundation for embodied agents. While pivotal, current machine learning modelsand in particular, vision-language models (VLMs)—still lag behind humans in this capacity (Liu et al., 2023b; Kamath et al., 2023; Fu et al., 2024), limiting applications in embodied domains such as robotic navigation and manipulation (Huang et al., 2023a;b; 2024b). To bridge the gap, extensive efforts have been devoted to enhancing the spatial intelligence of VLMs (Cheng et al., 2024; Qi et al., 2025; Zhang et al., 2025; Chen et al., 2024; Song et al., 2025; Zhou et al., 2025; Yuan et al., 2024).

Despite the remarkable progress, the evaluation of spatially intelligent VLMs remains limited. First, most existing benchmarks are static, adopting multiple-choice visual-question answering (VQA), though this facilitates automatic evaluation, the reliance on distractors renders them prone to biases. Moreover, VQA departs from practical scenarios, where VLM agents must proactively act upon given instructions in 3D rather than passively selecting an answer from a predefined set. Though more reliable real-world evaluations have been explored, the dependence on specific hardware and handcrafted tasks hinders their scalability and reproducibility (Yuan et al., 2024; Song et al., 2025).

Recently, some have eschewed discriminative VQA and proposed pointing, a generative evaluation methodology that requires models to locate the target object/space by generating points in 2D pixel space (Yuan et al., 2024; Zhou et al., 2025), but the execution phase that typically follows localization in robotics tasks has been overlooked or overly simplified. Others have attempted to address execution while circumventing the limitations of real-world evaluation using simulated environments (Liu et al., 2023a; Li et al., 2024b; Qi et al., 2025; Yang et al., 2025).Yet, both directions lack a systematic design of evaluation tasks that supports detailed analysis of spatial reasoning across different aspects (e.g., relationships and distances) and granularities (e.g., relative vs. precise distance).

Contact: yannzhao.ed@gmail.com, {dingwentao,lihongtao}@bigai.ai. †: Core Contributor. : Corresponding Author.

- Table 1: Comparisons of spatial-reasoning benchmarks. ‘Text Gen.’ and ‘Point Gen.’ indicate that models produce answers in natural language and 2D points, respectively. ‘Fully Gen.’ denotes that models generate positions and rotations in 3D. ‘Tool-Free’ means no external tools are used, thus assessing the intrinsic spatial reasoning of VLMs.

Clutter

Localization & Execution

Evaluation

PhysicallyGrounded

Diagnostic

Benchmark

Systematicity

Level Paradigm Tool-Free

image- and video-based

Blink (Fu et al., 2024) ✗ VQA ✓ ✗ ✗ ✗ high CV-Bench (Tong et al., 2024) ✗ VQA ✓ ✗ ✗ ✗ high VSI-Bench (Yang et al., 2024) ✗ VQA ✓ ✗ ✗ ✗ high WHERE2PLACE (Yuan et al., 2024) ✗ Point Gen. ✓ ✗ ✗ ✗ high SpatialVQA (Chen et al., 2024) ✗ VQA, Text Gen. ✓ ✗ ✗ ✗ high SpatialRGPT-Bench (Cheng et al., 2024) ✗ Text Gen. ✗ ✗ ✗ ✗ high RoboSpatial-Home (Song et al., 2025) ✗ VQA, Point Gen. ✓ ✗ ✗ ✗ high Point-Bench (Cheng et al., 2025) ✗ Point Gen. ✓ ✗ ✗ ✗ high

simulation-based

Open6DOR (Ding et al., 2024) ✓ VQA ✗ ✗ ✓ ✗ low EB-Manipulation (Yang et al., 2025) ✓ Fully Gen. ✗ ✗ ✓ ✗ low

ESPIRE (ours) ✓ Fully Gen. ✓ ✓ ✓ ✓ high

To address these limitations, we propose ESPIRE, a simulation-based benchmark for embodied spatial reasoning with physically-grounded VLMs. Since VLMs are inherently not trained to act, to adapt them for robotics tasks, we decompose each task into localization (which identifies manipulable targets) and execution (which performs the corresponding actions), and frame them as goal position and goal pose generation, respectively. This fully generative, unified evaluation paradigm extends passive spatial reasoning toward acting upon understanding, thus reducing the gap between evaluation and real-world deployment.

To serve our diagnostic purpose, we propose a systematic task design that enables assessment and analysis of the native spatial reasoning of VLMs across varying spatial aspects and granularities. We follow a hierarchical design philosophy, ensuring that the evaluation is spatial-centric and has a broad coverage. Specifically, we first identify three primary factors that characterize spatial reasoning: (1) spatial aspects, including attributes, relationships, distances, and orientations, (2) reference objects, including oriented and non-oriented, and (3) reference frames, including relative, intrinsic, and absolute. A particular configuration of these factors defines a context for spatial reasoning. For example, ‘place the book behind the picture frame’ requires reasoning about ‘positional relationship (behind)’ relative to an ‘oriented reference (picture frame)’ using the ‘intrinsic frame (attached to the picture frame)’. Within a given context, we curate tasks to examine reasoning across different granularities, e.g., fine-grained orientations in ‘grab a book to the 2 o’clock of the picture frame’ and precise distances in ‘grab a book within 1.2 meters of you.’ To the best of our knowledge, this systematic design supports the most comprehensive, fine-grained analysis that existing benchmarks lack.

We build ESPIRE on Isaac Sim (NVIDIA, 2025) that provides realistic physics simulation, and incorporate necessary measures to reduce sim-to-real gaps. ESPIRE offers a total of 148 spatialreasoning types for localization and covers typical pick and place actions, enabling a focus on VLM-oriented, embodied native spatial reasoning while maintaining sufficient challenges in toolfree execution. Combined with randomly sampled environments of varied clutter degrees, this provides broad coverage of spatial-centric reasoning and acting. To support scalable task generation, we represent task instructions in functional programs that can be executed on 3D scene graph representations of environment states and yield ground-truth targets.

We use ESPIRE to evaluate a diverse suite of VLMs, spanning proprietary, open-access, unified, and spatially-enhanced models. We find that VLMs perform much better in localization than in execution, indicating good passive spatial understanding but limited capacity for acting-oriented spatial reasoning. Among all spatial aspects, orientation reasoning poses the greatest challenge in both stages, suggesting a critical deficiency in grounding 3D rotational geometry. Overall, these findings highlight promising avenues for advancing the spatial cognition of VLMs. We emphasize that ESPIRE is not intended to replace real-world evaluation, but to complement it with a scalable, reproducible alternative that facilitates rapid, iterative model improvement.

In summary, our contributions are the following:

- • ESPIRE, a diagnostic benchmark for embodied spatial reasoning of VLMs in physically-grounded photorealistic environments.
- • A generative evaluation paradigm that unifies 3D localization and execution, bridging the gap between passive spatial understanding and acting-oriented spatial reasoning.
- • A systematic robotic task design that enables fine-grained diagnosis across diverse spatial reasoning contexts and granularities.
- • Experiments and analysis that quantify key bottlenecks in 3D rotational geometry and suggest future directions for enhancement.

- 2 RELATED WORK

Spatial reasoning with vision-language models. Extensive research has sought to boost the spatial intelligence of VLMs. Some rely on enhanced prompting mechanisms for improved 3D spatial reasoning (Ma et al., 2024; Liang et al., 2025), while many others adopt a data-centric method; in other words, they integrate 3D scene representations (e.g., depth maps and point clouds) into VLMs (Zhang et al., 2025; Qi et al., 2025). Meanwhile, many benchmarks have been proposed to evaluate their 2D and 3D spatial reasoning ability, including SpatialVQA (Chen et al., 2024), RoboSpatial-Home (Song et al., 2025), VSI-Bench (Yang et al., 2024), and many others (Liu et al., 2023b; Kamath et al., 2023; Cai et al., 2024; Fu et al., 2024; Cheng et al., 2024; Yuan et al., 2024; Chen et al., 2025; Zhang et al., 2025; Tong et al., 2024; Zhao et al., 2025). But these benchmarks are limited by their static nature and lack of systematic spatial-centric design. In addition, they predominantly adopt VQA-style evaluations, which are often prone to linguistic biases. In contrast, we propose a systematic task design and a unified generative paradigm, shifting the focus toward active, embodied evaluation.

Simulation-based evaluation through robotic tasks. Unlike human-assisted real-world evaluation, simulation-based approaches allow for more scalable and reproducible evaluation of robotics models, and have been widely used to assess robot policies in domains such as navigation and manipulation (Shridhar et al., 2020; Szot et al., 2021; Srivastava et al., 2022; Gu et al., 2023; James et al., 2020; Yu et al., 2020; Zeng et al., 2021; Mees et al., 2022; Ding et al., 2024). Due to the inherent limitations of simulators, substantial discrepancies exist between simulated observations and real-world observations. To bridge the gap, researchers have been improving physics engines and enhancing synthesis mechanisms to approximate real-world perceptions (Todorov et al., 2012; Xia et al., 2018; Anderson et al., 2018; NVIDIA, 2025). Though there have been simulated environments, such as LIBERO (Liu et al., 2023a), CALVIN (Mees et al., 2022), SIMPLER (Li et al., 2024b), and EmbodiedBench (Yang et al., 2025) for real-to-sim evaluation, they are limited in overly simplified scenes and tasks or reliance on external tools. In addition, none of them provides a systematic design of spatial-centric reasoning tasks and supports comprehensive diagnoses.

Foundation models for robotics manipulation. Foundation models, including pre-trained LLMs and VLMs, have been applied to robotic manipulation. Early work focuses primarily on task planning while relying on predefined primitives to achieve robot control (Ichter et al., 2022; Driess et al., 2023; Liang et al., 2023; Xie et al., 2023; Zhi et al., 2024). Recently, many have attempted to generate trajectories, i.e., sequences of poses, for motion planning (Huang et al., 2024a; 2023b; 2024b; Yuan et al., 2024; Qi et al., 2025) and devise agentic frameworks for reasoning and acting (GeminiRobotics-Team, 2025). Following the unified design philosophy, more recent efforts have focused on developing integrated vision-language-action models (VLAs) that can directly generate low-level action sequences as control policies (Brohan et al., 2023; Li et al., 2024a; Mees et al., 2024; Black et al., 2024; Ye et al., 2025; Bu et al., 2025; Wang et al., 2025), but their success hinges on the underlying spatial reasoning of their vision-language components, we focus on diagnosing VLMs to isolate and identify the specialized spatial inductive biases that are required to inform and improve future unified architectures.

6-DoF object rearrangement. 6-DoF object rearrangement involves predicting a goal state of an object that is described in SE(3) and satisfies the given instruction. With a motion planner, such a formulation enables zero-shot transfer of foundation models from perception to execution (Huang et al., 2023b; Kapelyukh et al., 2024). The approaches to 6-DoF tasks can be roughly divided into

[Figure 1]

[Figure 2]

[Figure 3]

distant reference w/o intrinsic frame

distant reference w/ intrinsic frame

behind

front

near reference w/ intrinsic frame

[Figure 4]

##### Tabletop Scene

above

below

near reference w/o intrinsic frame

varying-sized, -oriented manipulable books

##### Shelf Scene

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

Grab a book at the 12 o'clock of the cheval mirror.

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

Place the book in a slot immediately to the left of the alarm clock.

Localization Execution 1 Execution 2 Done

Figure 1: ESPIRE: a simulated physical world. Top: the spatial world of ESPIRE covers key factors of spatial reasoning like spatial aspects (e.g., relationship and distance), reference frames, reference objects (§4.1). It features a tabletop scene for pick tasks and a shelf scene for place tasks (§4.2) and supports reasoning at varying granularities (see Table 10 in Appendix A.2). Bottom: example ESPIRE tasks that all inherently rely on spatial reasoning.

generative- and discriminative-based. Generative methods solve for a goal translation and rotation of a directional vector under certain constraints (Huang et al., 2024a;b), while discriminative approaches generate random candidates and use a critic to filter and select the best goal pose (Ding et al., 2024; Kapelyukh et al., 2024). We follow the generative paradigm and prompt VLMs to generate a goal pose and ground it in the simulated physical world.

- 3 SPATIAL-CENTRIC EVALUATION OF EMBODIED VLMS

We propose evaluating the spatial cognition of VLMs through robotics tasks situated in a simulated physical world, narrowing the gap between evaluation and real-world deployment. To adapt VLMs for robotics tasks, we decompose each task into two sequential subtasks: localization and execution, formulate them as generative tasks, and ensure that spatial reasoning is the key factor.

- • Localization refers to locating a target that is specified in a given instruction from the paired scene, such as the ‘book’ in ‘pick up the farthest book’ and the ‘empty spot’ in ‘place the book in an empty spot’. We follow Yuan et al. (2024) and Zhou et al. (2025) and formulate it as a pointing task that produces 2D coordinates on scene images.

Evaluation Metric. We measure model performance using accuracy, defined as the fraction of correct localizations. Unlike discriminative VQA-style evaluations that rely on distractors for automatic metrics, our generative formulation allows for directly comparing the predicted point against the target segmentation mask.

- • Execution follows the localization stage to execute actions (e.g., pick or place) in the physically grounded environment. Since VLMs cannot directly produce low-level control actions, we simplify execution as a 6-DoF task that predicts the goal pose, including goal position and orientation prediction, in SE(3). We again formulate goal position prediction as a pointing task.

Evaluation Metric. We measure model performance using acceptance rate, defined as the fraction of physically achieved poses. The acceptability of a predicted pose is assessed by a motion planner like cuRobo (Sundaralingam et al., 2023), making VLMs physically grounded.

In both tasks, native spatial reasoning is inherently needed since VLMs are required to generate positions and orientations in 3D, without relying on external tools. The shared pointing formulation between localization and execution further bridges spatial reasoning for understanding and for acting.

- 4 THE ESPIRE BENCHMARK

We propose ESPIRE, a simulated environment that provides a suite of robotics tasks for diagnosing spatial-centric reasoning (see Figure 1). We design ESPIRE systematically both in instructions (§4.1) and environments (§4.2), ensuring a broad coverage of spatial reasoning scenarios, enabling scalable robotic task generation (§4.3), and supporting targeted analysis across contexts and granularities.

- 4.1 SPATIAL REASONING TASKS

Task specification. We group spatial reasoning tasks into four broad classes by the spatial aspects they require to reason about: relationships, distances, attributes (e.g., dimensions and volumes), and orientations. A spatial reasoning task typically involves describing an object in relation to another (e.g., ‘grab the book to your left’), thus relying on a frame of reference. Following Levinson (2003), we consider three types of reference frames: relative, intrinsic, and absolute frames. The choice of reference frame depends on the reference object, e.g.. intrinsic-oriented objects like ‘picture frame’ that have a clear front face naturally support intrinsic frames, whereas non-oriented objects like ‘sphere ball’ do not. Moreover, the reference frame may vary with linguistic specifications, e.g., ‘pick up a book on the left of the picture frame’ exhibits ambiguity since both a relative frame and an intrinsic frame can be used, but attaching the clause ‘relative to the picture frame’s front’ makes the intrinsic frame the only valid interpretation.

To disentangle this complexity, we identify three key factors that characterize spatial reasoning: spatial aspect (S), reference frame (F), and reference object (O); we define their combination C = (S,F,O) as the task specification. A particular configuration of these factors specifies a context for spatial reasoning. For example, c = (relationship,intrinsic,table) requires using the intrinsic frame of the table to carry out relationship reasoning; an instance of it can be ‘grab a book on the left of the table.’ This disentanglement lets us focus on designing tasks that target reasoning at varying granularities like left, leftmost, second leftmost, and to your 11 o’clock.

Instruction representation. We associate each task instruction with a 3-tuple T = (C,A,P), where C denotes the task specification, A ∈ {pick,place} represents execution, and P indicates localization. We represent P as a functional program (Johnson et al., 2017) that can be evaluated on the 3D scene graph representation G of a given environment state and produces a list of valid answers, i.e., objects to be manipulated or spaces to be filled. Crucially, A functional program is composed of atomic functions and defines a reasoning chain, such as finding a specific object unique(filter(O,G)) and querying the objects to its left filterRel(left,unique(filter(O,G))). This enables flexible control of the task complexity by varying the number of reasoning hops.

Instruction families. We define an instruction family on top of a task T = (C,A,P) by associating it with a set of task templates that represent different linguistic expressions of the functional program P. Supposing C = (distance,viewer,intrinsic), A = ‘Pick’, and a template ‘[A] a book among the books [R] you’, we can create an instruction, which queries the distance between a book and the viewer, by binding the variable [R] with a type of distance reasoning (e.g., Closest or Furthest). Using the same variable [R], the functional program P can be formed as:

filterDist([R],filter(book,G),viewer)

We curate a total of 148 spatial-reasoning task types, distributed across 65 instruction families, including 31 ‘pick’ instruction families and 34 ‘place’ instruction families. For each instruction family, we manually write 3-4 templates to enhance linguistic diversity. Though functional programs enable multi-hop compositional reasoning, we limit reasoning up to 3 hops, as our primary focus is on spatial rather than compositional reasoning.1 In practice, we find that a small number of spatial reasoning hops already poses challenges for existing multimodal foundation models.

1Nonetheless, ESPIRE can be readily extended by increasing the number of spatial reasoning hops.

- 4.2 SIMULATION ENVIRONMENT

We simulate two task environments in ESPIRE: tabletop and shelf scenes. Both are constructed systematically using a diverse array of photorealistic objects and various spatial layouts and environmental factors like lighting and clutter. This design ensures that our environments provide a comprehensive instantiation of the task specification C, yielding diverse instances that challenge model reasoning across multiple levels of granularity (refer to Appendix A.2 for detailed scene configurations).

Environment representation and generation. We initialize each environment from a random state, which is represented by a 3D scene graph that consists of nodes as objects and edges as spatial relationships. All objects are annotated with ground-truth information, including sizes, dimensions, and poses relative to a predefined absolute reference frame. We generate the initial state of an environment by sampling a random 3D scene graph and rendering it in Isaac Sim (NVIDIA, 2025), ensuring that the environment is physically valid. We adjust the minimum margin of objects and the dimensions of shelf slots; this mitigates the visual ambiguity of spatial aspects and accommodates sufficient, physically feasible tasks in the environment. The Franka robot is initialized in a random pose. We equip it with an on-wrist camera that provides an egocentric view and supplement it with two fixed-position cameras that provide global views of the tabletop and shelf scenes, respectively (referred to as world views). To increase variety and realism, we add external lights. We randomly sample and initialize the positions and orientations of all cameras and external lights.

Reducing the real-to-sim visual gaps. Visual gaps mainly arise from distribution shifts in texture, material, lighting, and camera configurations. Instead of performing complex visual-matching mitigation as in SimplerEnv (Li et al., 2024b), we employ a more scalable strategy that focuses on enhancing the diversity of the environment: we use annotated 3D assets with realistic textures and tune their sizes to reflect their real-world counterparts. For essential background assets like the tabletop and shelf, we randomly assign textures derived from real-world materials. Combined with randomization in lighting and camera poses, this produces a diverse and visually realistic set of environments (see details in Appendix A.2 and a discussion on sim-to-real relevance in Appendix A.3).

- 4.3 SIMULATION TASKS

A simulation task is defined by a pair of an environment state and a task instruction. We generate ‘pick’ and ‘place’ tasks sequentially. First, we sample and render an environment. The Franka robot is always initialized in a position suitable for performing ‘pick’ tasks, so we start with ‘pick’ task generation, and ‘place’ task generation follows the same procedure. For each variable in a given instruction family, after sampling a random type, we perform value filtering. This is particularly useful for the reference-object variable, as not all reference objects appear in the task space visible from the world view. Once all variables are bound and instantiated, we obtain the final functional program and execute it on the 3D scene graph representation of the visible portion of the environment state. The yielded answers are further verified using a motion planner. We only retain those that correspond to feasible manipulations.2 Finally, we randomly select a task template from the given task family and instantiate it into a natural language instruction.

- 5 EXPERIMENTS

- 5.1 EXPERIMENTAL SETUPS

Evaluated models. We consider a diverse range of multimodal foundation models, including proprietary VLMs like Gemini2.5-Pro (Team et al., 2025), public general-purpose VLMs like instruction-tuned Qwen3-VL (Bai et al., 2025) and InternVL3 (Zhu et al., 2025), and spatial-reasoning enhanced VLMs like RoboBrain2.0 (RoboBrain-Team et al., 2025).

2We assume the robot can move freely in 3D, with both locomotion across the ground plane and vertical motion along the global up-axis. This relaxation facilitates reliable execution when using VLMs and a large task space that broadens the coverage of spatial reasoning scenarios.

- Table 2: The localization accuracy (%), acceptance rate (%) in execution, and overall task success rate (%) across different VLMs.

Pick Place accuracy acceptance success accuracy acceptance success

Models

w/o reflection

Gemini2.5-Pro 57.72 63.93 34.06 50.61 28.36 5.68 InternVL3-78B 28.31 63.01 17.26 23.66 40.94 9.67 RoboBrain2.0-7B 57.72 18.81 10.87 50.70 15.68 8.64 Qwen3-VL-30B-A3B 54.43 62.56 32.15 45.54 43.47 20.00 Qwen3-VL-8B 47.03 63.20 29.32 35.71 37.31 12.41 Qwen3-VL-235B-A22B 51.96 52.79 26.76 47.42 41.22 19.34

w/ reflection

Qwen3-VL-30B-A3B 54.52 27.85 17.08 51.92 23.94 13.80 Qwen3-VL-8B 58.63 24.38 15.07 54.08 12.02 6.67 Qwen3-VL-235B-A22B 64.29 36.71 23.20 59.72 25.45 15.40

Evaluation tasks. Each task family is paired with at least 15 different scenes, leading to around 15 trials on average. We define the difficulty of a task as the complexity of the accompanying scene. Specifically, we categorize the tabletop and shelf tasks into three difficulty levels: easy, medium, and hard. For tabletop tasks, the three levels correspond to scenes that contain 1-2, 3-5, and 6-8 books on the table, respectively.3 For shelf tasks, difficulty is defined as the fullness of the associated shelf: easy, medium, and hard correspond to shelves where one-third, two-thirds, and all slots are occupied, respectively (refer to Table 14 in Appendix A.5 for illustrations).

Evaluation settings. Our evaluation suite offers a total of 2,220 tasks, consisting of 1,095 pick tasks and 1,125 place tasks. We limit the number of attempts to 3 for localization and 5 for execution. If localization fails, we randomly select a gold target for execution; otherwise, we use the target localized by the model for execution. We consider non-reflection and reflection settings. In the non-reflection setting, the initial observation is provided by the world-view, while all subsequent observations are obtained from the ego-view. In the reflection setting, the model additionally receives as input its reflections from the previous failed attempt (refer to Algorithms 2 and 3 in Appendix A.4). To enhance reflection in execution, both the world-view and the ego-view are provided. We prompt models to output a 2D point in pixel space while providing them with ground-truth depth. Depending on the settings, we may provide ground-truth rotations of pitch, yaw, and roll or prompt models to generate them directly.

- 5.2 MAIN RESULTS

In general, proprietary VLMs like Gemini2.5-Pro show the strongest performance under most metrics, while public VLMs like the Qwen3-VL series is narrowing the gaps and even outperform

- Gemini2.4-Pro in execution on place tasks. Interestingly, larger models do not necessarily lead to better performance, e.g., Qwen3-VL-30B with 3B activated parameters outperforms Qwen3VL-8B and Qwen3-VL-235B (w/ 22B activated parameters). Moreover, all models demonstrate decent localization accuracy except InternVL3-78B, which has an accuracy below 30% (see Table 2), possibly because the native multimodal pre-training adopted by InternVL3-78B for aligning vision and language is not as effective as widely-used stage-by-stage alignment learning (Liu et al., 2023c). Despite that, it is surprising to see that InternVL3-78B performs much better (e.g., >40%) in execution than in localization. While RoboBrain2.0-7B achieves impressive localization performance, it is mostly because its post-training involves extensive generic spatial reasoning tasks, but, unfortunately, this does not transfer to improved performance in execution (e.g., <20%) that requires acting-oriented spatial reasoning about 3D rotational geometry.

‘Place’ is generally harder than ‘pick.’ Compared with pick tasks, place tasks impose much stricter acceptance conditions. Specifically, when predicting a pose for the placement of a book, the model needs to consider additional constraints of the target space, especially when the target space

3The book number best correlates with task complexity, but overall complexity is driven by instructions and environmental factors like object number and pose, light, and texture.

[Figure 13]

Figure 2: Localization performance across spatial aspects and granularities on pick tasks.

is partially occupied. Moreover, the model usually suffers much more from occlusion when it gets closer to the target space, making it harder to predict a relatively-center point of the space or recover from a non-ideal position, whereas in pick tasks, the model only needs to align with any one of all graspable faces (e.g., the spine or the top edge), and it can optionally move to a better position to facilitate pose prediction (refer to our qualitative studies in Table 6).

Reflection is helpful for localization but does not necessarily help with execution. While reflection improves localization performance of all Qwen3-VL models (see Table 1), it does not yield a comparable improvement in execution; conversely, it degrades execution performance. This is likely because strong 3D rotation understanding is the key factor for execution and forms the foundation for reflection; however, as we will see in an analysis of rotation predictions (see Table 7), current VLMs are weak in this capacity, suggesting that future work is well-suited for curating rotation-reasoning data for fine-tuning.

- 5.3 ANALYSIS

The systematic design of ESPIRE enables finegrained analysis of model behavior. We demonstrate this by examining spatial reasoning performance across spatial aspects and task difficulty levels, and by analyzing behavior during successful task execution.

Table 3: Localization accuracy (%) across four primary spatial aspects.

Accuracy attribute distance orientation relationship

Models

Pick

Gemini2.5-Pro 56.00 53.89 63.81 56.52 InternVL3-78B 29.33 21.67 33.65 30.14 RoboBrain2.0-7B 61.33 56.11 60.95 55.65 Qwen3-VL-30B-A3B 57.33 50.83 57.14 55.07 Qwen3-VL-8B 42.67 40.28 53.97 48.70 Qwen3-VL-235B-A22B 49.33 49.44 54.60 52.75 AVERAGE 49.33 45.37 54.02 49.81

Localization performance across spatial aspects. We group results by spatial aspects (see

- Table 3 and Figure 2). Overall, all models perform worse on ‘distance’ than on the other spatial aspects, across pick and place tasks, indicating that current VLMs lack the capacity for precise distance understanding. Among them,

Place

Gemini2.5-Pro 53.33 46.09 48.75 55.06 InternVL3-78B 28.00 16.23 21.25 30.62 RoboBrain2.0-7B 69.33 45.80 47.08 53.58 Qwen3-VL-30B-A3B 48.00 38.55 39.58 54.57 Qwen3-VL-8B 44.00 28.41 31.67 42.82 Qwen3-VL-235B-A22B 62.67 37.68 46.26 53.58 AVERAGE 47.20 33.33 37.17 47.03

- Gemini2.5-Pro and RoboBrain2.0-7B exhibit relatively stronger overall performance while showing smaller performance variations across spatial aspects, likely because they have been specifically fine-tuned on related spatial reasoning tasks; this is explicitly the case for RoboBrain2.0-7B.

- Table 5: The average number of attempts to succeed in localization and execution, and average distance (meter) between the target and end-effector upon execution success and before execution success. ‘Rank’ indicates model ranking in execution.

Dist. at success

Dist. before success

Models #Localization #Move

Rank Pick

Gemini2.5-Pro 1.20 2.54 0.07 0.47 1 InternVL3-78B 1.05 2.56 0.05 0.48 3 RoboBrain2.0-7B 1.36 2.54 0.05 0.50 6 Qwen3-VL-30B-A3B 1.16 2.49 0.06 0.38 4 Qwen3-VL-8B 1.17 2.41 0.05 0.42 2 Qwen3-VL-235B-A22B 1.18 2.53 0.05 0.40 5

Place

Gemini2.5-Pro 1.42 3.27 0.26 0.75 5 InternVL3-78B 1.08 2.07 0.24 0.97 3 RoboBrain2.0-7B 1.59 2.98 0.24 0.85 6 Qwen3-VL-30B-A3B 1.33 2.12 0.24 0.95 1 Qwen3-VL-8B 1.30 2.10 0.24 0.97 4 Qwen3-VL-235B-A22B 1.28 2.16 0.23 0.91 2

Model performance across task difficulty levels. We further group results by task difficulty levels (see Table 4). Unsurprisingly, on pick tasks, most models demonstrate a decrease in both localization execution with increasing task difficulty, except InternVL3-78B and RoboBrain2.0-7B that, in some cases, perform slightly better on harder tasks. Similarly, on place tasks, both the localization performance and the execution performance negatively correlate with the task difficulty. Still, there are exceptions like InternVL3-78B and Qwen3-VL-30B-A3B.

Prerequisites for successful execution. Next, we analyze the prerequisites that are strongly associated with successful execution. To this end, we compute the average number of attempts used to achieve successful localization and execution, and the average distances between the target and the end-effector upon execution success and before execution success (see Table 5). Though there is no clear correlation between execution success and pre-success distance, we find that InternVL3-78B and Qwen3-VL-30B-A3B, which are relatively better at execution, tend to be far-sighted (e.g., w/ an average distance of 48cm) and near-sighted (e.g., w/ an average distance of 38cm), respectively. Interestingly, the pre-success distance in place tasks is usually twice that in pick tasks, presumably because, in place tasks, the robot needs to stay reasonably far away from the target space to mitigate occlusion. Moreover, in place tasks, strong models like the Qwen3-VL series often require a moderate number of moves; that is, they tend to try multiple times (i.e., around 2.1) before making the final successful execution. In contrast, models that try much more times (i.e., around 3) are usually weaker in execution, e.g., RoboBrain2.0-7B fails spectacularly because it struggles in acting-oriented spatial reasoning.

Table 4: Performance across difficulty levels.

Accuracy (%) Acceptance (%) easy medium hard easy medium hard

Models

Pick Gemini2.5-Pro 60.78 60.98 52.04 70.96 60.98 60.71

- InternVL3-78B 24.85 29.00 30.61 60.78 65.85 62.24 RoboBrain2.0-7B 62.57 56.10 55.10 21.56 15.72 19.39 Qwen3-VL-30B-A3B 62.57 53.39 48.87 65.87 66.67 55.87 Qwen3-VL-8B 58.08 42.82 41.58 64.97 63.96 60.97 Qwen3-VL-235B-A22B 59.58 52.57 44.90 58.68 55.56 45.15

Place Gemini2.5-Pro 57.46 48.21 46.11 36.06 28.37 20.46

- InternVL3-78B 25.35 22.31 23.34 51.55 39.12 31.99 RoboBrain2.0-7B 52.68 52.62 46.69 18.31 16.80 11.82 Qwen3-VL-30B-A3B 47.61 44.08 44.96 52.39 43.80 34.01 Qwen3-VL-8B 37.18 36.74 33.14 42.25 40.06 29.39 Qwen3-VL-235B-A22B 51.83 48.48 41.79 48.17 43.53 31.70

Apart from the above quantitative analysis, we present qualitative analysis of both successful runs and failed runs in Table 6.

- 5.4 ABLATION OF ROTATION PREDICTION

At the core of execution lies the prediction of rotations along the pitch, roll, and yaw axes. This reflects the model’s capability for 3D geometric reasoning and its understanding of object affordances,

- as the predicted rotations are further composed into a goal pose for execution. To better understand

- Table 6: Qualitative analysis. We categorize intermediate executions into following six types: T1 denotes a grasp-favorable viewpoint; T2 denotes a grasp-infeasible viewpoint; T3 denotes manipulator occlusion; T4 denotes object occlusion; T5 denotes unrecognizable target; and T6 denotes physicallyachievable execution.

Model Step 1 Step 2 Step 3 Step 4 Step 5 Step 6

Find a book at 12 o’clock of the cheval mirror from the table, and grab it.

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

- T2 T2 T2 T1 T6 Execution

[Figure 21]

235B

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

- T2 T2 T2 T2 T2 Execution

Place the book in the shelf position (row 1, column 5).

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

- T5 T6 T4 T6 Execution

[Figure 34]

30B

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

- T6 T3 T4 T6 T6 Execution

the intrinsic capability of VLMs for rotation prediction, we ablate the set of rotation axes to be predicted using Qwen3-VL-235B-A22B.

Specifically, rather than using the ground-truth angles derived from the predicted grasping face of the target book/space, we instruct the VLM to directly generate rotation angles for pitch, yaw, and roll. We randomly sample 110 pick and 106 place tasks from the test suite for this ablation study. Since most place tasks impose no constraints on the final pose, we additionally include 60 place tasks with explicit pose constraints (e.g., ‘place the book

Table 7: Acceptance rate with rotation axes generated by Qwen3-VL-235BA22B. Unchecked axes indicate that ground-truth rotations are used. For place tasks, results are reported on tasks with and without explicit pose constraints (C).

- at a tilt of 60 degrees.’).

Place (%)

Pitch Yaw Roll Pick (%)

Interestingly, pitch and roll appear to be the key factors for pick tasks and constrained place tasks, respectively (see Table 7). We conjecture that pick tasks require the model to select a feasible grasping face, which largely depends on the pitch axis, whereas constrained place tasks require the model to determine a deviation from the upright direction, primarily governed by the roll axis. As expected, execution becomes harder as more axes need to be predicted. In particular, the pitch-yaw combination adversely affects pick the most, while yaw-roll has the largest impact on constrained place tasks.

w/o C w/ C

52.73 37.74 43.33 ✓ 20.91 26.42 26.67 ✓ 28.18 29.25 25.00

✓ 30.91 35.85 23.33 ✓ ✓ 4.55 23.58 25.00 ✓ ✓ 13.64 33.02 16.67

✓ ✓ 11.82 32.08 10.00 ✓ ✓ ✓ 3.64 24.53 16.00

- 5.5 HUMAN STUDY

As discussed in Section 4.1, spatial reasoning tasks may exhibit ambiguity when the intended reference frame is not explicitly specified and must be inferred. For example, given the instruction

‘grab a book to the left of the picture frame,’ an agent must determine whether to interpret the relation using the intrinsic frame or the relative frame. To investigate to what extent VLMs exhibit frame preferences similar to humans, we use near oriented objects, distant oriented objects, and the table as the reference, construct 91 pick tasks involving ambiguous frames, and collect responses from five human participants. Then, we measure human-model agreement by computing the Spearman’s rank correlation.

Among the three types of oriented reference objects, humans and models show strong agreement when the table or distant objects are used as references (i.e., w/ high positive correlations above 0.8) but disagree (i.e., w/ negative correlations) when the reference is a near object such as an alarm clock. In such cases, we find that humans tend to prefer the intrinsic frame of the reference object while models favor the relative frame. We hypothesize that when the reference and target objects (i.e., books) are of comparable size, humans perceive the reference as an oriented object with salient geometric cues, making its intrinsic frame more accessible. In contrast, VLMs appear to struggle with object-centric orientation inference and therefore default to the relative frame.

Table 8: Agreement of humans and models on reference frames. References are categorized into table, near, and distant objects.

Model Near Obj. Distant Obj. Table

Gemini2.5-Pro -0.573 ± 0.634 0.8 ± 0.274 1.0 ± 0.0 RoboBrain2.0-7B -0.674 ± 0.242 0.8 ± 0.274 1.0 ± 0.0 Qwen3-VL-30B-A3B -0.100 ± 0.652 0.8 ± 0.274 1.0 ± 0.0 Qwen3-VL-8B -0.674 ± 0.242 0.8 ± 0.274 1.0 ± 0.0 Qwen3-VL-235B-A22B -0.573 ± 0.634 0.8 ± 0.274 1.0 ± 0.0

- 5.6 EFFICIENCY OF ESPIRE

Our analysis regarding the running time of ESPIRE reveals two primary sources of latency: API calls and model inference. API response time is largely affected by network stability, whereas the model inference time is determined by the model size and the hardware used for deployment. Taking RoboBrain2.0-7B (RoboBrain-Team et al., 2025), when running on an RTX 4090 machine, a single inference takes an average of 9.25 seconds. Another source of latency comes from execution that involves motion planning and environment update. Specifically, the average time for executing a move request is about 18.12 seconds in our experiments on a workstation equipped with an NVIDIA RTX 4090 GPU.

- 6 DISCUSSION AND FUTURE WORK

ESPIRE is the first simulated physical environment designed for the diagnostic evaluation of spatial cognition in VLMs, featuring spatial-centric robotic tasks that are explicitly designed to be scalable and diverse. To evaluate VLMs that cannot directly produce low-level control actions, we have reformulated robotic tasks into localization and execution. While future VLAs are supposed to integrate the two phases, we deliberately prioritize diagnosis over integration. This design choice is further motivated by existing agentic frameworks that decouple reasoning and acting, using VLMs for high-level spatial reasoning and VLAs or controllers for action execution (Gemini-Robotics-Team, 2025). By isolating the reasoning stage, our framework provides a ‘microscope’ to identify where spatial reasoning chains break, offering a concrete roadmap for the specialized spatial inductive biases that future architectures will require.

A limitation of ESPIRE is: it is restricted to indoor scenes. Despite our systematic design, it does not cover spatial reasoning scenarios that arise only outdoors, such as reasoning with larger units of measure (e.g., kilometer), reasoning with larger-sized reference objects (e.g., trees), and reasoning using the global reference frame (e.g., south or east). Nonetheless, ESPIRE readily supports such extensions, for example, by making outdoor reference objects visible through glass walls.

Beyond that, ESPIRE opens several new avenues for the development and analysis of spatially intelligent VLMs. For example, ESPIRE allows for designing long-horizon tasks that require multistep spatial reasoning, leading to many interesting model analyses, including the modeling of dependencies between reasoning steps and the role of memory in long-horizon spatial reasoning. Moreover, since ‘pick’ and ‘place’ tasks typically occur sequentially in robotics, but are performed in different workspaces in ESPIRE, it is well-suited for extending it to evaluate mobile manipulation.

- 7 CONCLUSION

We have presented ESPIRE, a simulated environment that provides an evaluation suite for embodied spatial reasoning with vision-language models. ESPIRE evaluates VLMs on robotic tasks in a physically grounded setting, thus mitigating the gap between evaluation and practical deployment. By breaking down each task into localization and execution, ESPIRE provides a unified evaluation of passive spatial reasoning and action-oriented spatial reasoning. We systematically design ESPIRE to simulate a diverse range of spatial reasoning scenarios, enabling a comprehensive analysis across spatial aspects and at multiple levels of granularity. Our experimental results and analysis reveal future directions for enhancing VLMs in spatial reasoning.

- 8 ACKNOWLEDGEMENTS

Yanpeng Zhao acknowledges the support of the National Natural Science Foundation of China

(12574467).

REFERENCES

Peter Anderson, Qi Wu, Damien Teney, Jake Bruce, Mark Johnson, Niko Sünderhauf, Ian Reid, Stephen Gould, and Anton van den Hengel. Vision-and-language navigation: Interpreting visually-grounded navigation instructions in real environments. In 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 3674–3683, 2018. doi: 10.1109/CVPR.2018.00387.

Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-vl technical report, 2025. URL https://arxiv.org/abs/2502.13923.

Kevin Black, Noah Brown, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, Lachy Groom, Karol Hausman, Brian Ichter, Szymon Jakubczak, Tim Jones, Liyiming Ke, Sergey Levine, Adrian Li-Bell, Mohith Mothukuri, Suraj Nair, Karl Pertsch, Lucy Xiaoyang Shi, James Tanner, Quan Vuong, Anna Walling, Haohuan Wang, and Ury Zhilinsky. π0: A vision-language-action flow model for general robot control, 2024. URL https://arxiv.org/abs/2410.24164.

Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Joseph Dabis, Chelsea Finn, Keerthana Gopalakrishnan, Karol Hausman, Alexander Herzog, Jasmine Hsu, Julian Ibarz, Brian Ichter, Alex Irpan, Tomas Jackson, Sally Jesmonth, Nikhil J. Joshi, Ryan Julian, Dmitry Kalashnikov, Yuheng Kuang, Isabel Leal, Kuang-Huei Lee, Sergey Levine, Yao Lu, Utsav Malla, Deeksha Manjunath, Igor Mordatch, Ofir Nachum, Carolina Parada, Jodilyn Peralta, Emily Perez, Karl Pertsch, Jornell Quiambao, Kanishka Rao, Michael S. Ryoo, Grecia Salazar, Pannag R. Sanketi, Kevin Sayed, Jaspiar Singh, Sumedh Sontakke, Austin Stone, Clayton Tan, Huong Tran, Vincent Vanhoucke, Steve Vega, Quan Vuong, Fei Xia, Ted Xiao, Peng Xu, Sichun Xu, Tianhe Yu, and Brianna Zitkovich. Rt-1: Robotics transformer for real-world control at scale. In Robotics: Science and Systems, 2023. URL https://doi.org/10.15607/RSS.2023.XIX.025.

Qingwen Bu, Yanting Yang, Jisong Cai, Shenyuan Gao, Guanghui Ren, Maoqing Yao, Ping Luo, and Hongyang Li. Univla: Learning to act anywhere with task-centric latent actions, 2025. URL https://arxiv.org/abs/2505. 06111.

Wenxiao Cai, Yaroslav Ponomarenko, Jianhao Yuan, Xiaoqi Li, Wankou Yang, Hao Dong, and Bo Zhao. Spatialbot: Precise spatial understanding with vision language models. CoRR, abs/2406.13642, 2024. URL https://doi.org/10.48550/arXiv.2406.13642.

Boyuan Chen, Zhuo Xu, Sean Kirmani, Brain Ichter, Dorsa Sadigh, Leonidas Guibas, and Fei Xia. Spatialvlm: Endowing vision-language models with spatial reasoning capabilities. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 14455–14465, June 2024.

Kaiyuan Chen, Shuangyu Xie, Zehan Ma, Pannag R Sanketi, and Ken Goldberg. Robo2vlm: Visual question answering from large-scale in-the-wild robot manipulation datasets, 2025. URL https://arxiv.org/abs/2505. 15517.

An-Chieh Cheng, Hongxu Yin, Yang Fu, Qiushan Guo, Ruihan Yang, Jan Kautz, Xiaolong Wang, and Sifei Liu. SpatialRGPT: Grounded spatial reasoning in vision-language models. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024. URL https://openreview.net/forum?id=JKEIYQUSUc.

Long Cheng, Jiafei Duan, Yi Ru Wang, Haoquan Fang, Boyang Li, Yushan Huang, Elvis Wang, Ainaz Eftekhar, Jason Lee, Wentao Yuan, Rose Hendrix, Noah A. Smith, Fei Xia, Dieter Fox, and Ranjay Krishna. Pointarena: Probing multimodal grounding through language-guided pointing, 2025. URL https://arxiv.org/abs/2505. 09990.

Yufei Ding, Haoran Geng, Chaoyi Xu, Xiaomeng Fang, Jiazhao Zhang, Songlin Wei, Qiyu Dai, Zhizheng Zhang, and He Wang. Open6dor: Benchmarking open-instruction 6-dof object rearrangement and a vlmbased approach. In 2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pp. 7359–7366, 2024. doi: 10.1109/IROS58592.2024.10802733.

Danny Driess, Fei Xia, Mehdi S. M. Sajjadi, Corey Lynch, Aakanksha Chowdhery, Brian Ichter, Ayzaan Wahid, Jonathan Tompson, Quan Vuong, Tianhe Yu, Wenlong Huang, Yevgen Chebotar, Pierre Sermanet, Daniel Duckworth, Sergey Levine, Vincent Vanhoucke, Karol Hausman, Marc Toussaint, Klaus Greff, Andy Zeng, Igor Mordatch, and Pete Florence. PaLM-e: An embodied multimodal language model. In Andreas Krause, Emma Brunskill, Kyunghyun Cho, Barbara Engelhardt, Sivan Sabato, and Jonathan Scarlett (eds.), Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, pp. 8469–8488. PMLR, 23–29 Jul 2023. URL https://proceedings.mlr.press/ v202/driess23a.html.

Xingyu Fu, Yushi Hu, Bangzheng Li, Yu Feng, Haoyu Wang, Xudong Lin, Dan Roth, Noah A. Smith, WeiChiu Ma, and Ranjay Krishna. Blink: Multimodal large language models can see but not perceive. In Computer Vision – ECCV 2024: 18th European Conference, Milan, Italy, September 29–October 4, 2024, Proceedings, Part XXIII, pp. 148–166, Berlin, Heidelberg, 2024. Springer-Verlag. ISBN 978-3-031-73336-9. doi: 10.1007/978-3-031-73337-6_9. URL https://doi.org/10.1007/978-3-031-73337-6_9.

Gemini-Robotics-Team. Gemini robotics 1.5: Pushing the frontier of generalist robots with advanced embodied reasoning, thinking, and motion transfer, 2025. URL https://deepmind.google/discover/blog/ gemini-robotics-15-brings-ai-agents-into-the-physical-world/.

Jiayuan Gu, Fanbo Xiang, Xuanlin Li, Zhan Ling, Xiqiang Liu, Tongzhou Mu, Yihe Tang, Stone Tao, Xinyue Wei, Yunchao Yao, Xiaodi Yuan, Pengwei Xie, Zhiao Huang, Rui Chen, and Hao Su. Maniskill2: A unified benchmark for generalizable manipulation skills. In The Eleventh International Conference on Learning Representations, 2023. URL https://openreview.net/forum?id=b_CQDy9vrD1.

Chenguang Huang, Oier Mees, Andy Zeng, and Wolfram Burgard. Visual language maps for robot navigation. In 2023 IEEE International Conference on Robotics and Automation (ICRA), pp. 10608–10615, 2023a. doi: 10.1109/ICRA48891.2023.10160969.

Haoxu Huang, Fanqi Lin, Yingdong Hu, Shengjie Wang, and Yang Gao. Copa: General robotic manipulation through spatial constraints of parts with foundation models. In 2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pp. 9488–9495, 2024a. doi: 10.1109/IROS58592.2024.10801352.

Wenlong Huang, Chen Wang, Ruohan Zhang, Yunzhu Li, Jiajun Wu, and Li Fei-Fei. Voxposer: Composable 3d value maps for robotic manipulation with language models. In Jie Tan, Marc Toussaint, and Kourosh Darvish (eds.), Proceedings of The 7th Conference on Robot Learning, volume 229 of Proceedings of Machine Learning Research, pp. 540–562. PMLR, 06–09 Nov 2023b. URL https://proceedings.mlr.press/ v229/huang23b.html.

Wenlong Huang, Chen Wang, Yunzhu Li, Ruohan Zhang, and Li Fei-Fei. Rekep: Spatio-temporal reasoning of relational keypoint constraints for robotic manipulation. In 8th Annual Conference on Robot Learning, 2024b. URL https://openreview.net/forum?id=9iG3SEbMnL.

Brian Ichter, Anthony Brohan, Yevgen Chebotar, Chelsea Finn, Karol Hausman, Alexander Herzog, Daniel Ho, Julian Ibarz, Alex Irpan, Eric Jang, Ryan Julian, Dmitry Kalashnikov, Sergey Levine, Yao Lu, Carolina Parada, Kanishka Rao, Pierre Sermanet, Alexander T Toshev, Vincent Vanhoucke, Fei Xia, Ted Xiao, Peng Xu, Mengyuan Yan, Noah Brown, Michael Ahn, Omar Cortes, Nicolas Sievers, Clayton Tan, Sichun Xu, Diego Reyes, Jarek Rettinghouse, Jornell Quiambao, Peter Pastor, Linda Luu, Kuang-Huei Lee, Yuheng Kuang, Sally Jesmonth, Kyle Jeffrey, Rosario Jauregui Ruano, Jasmine Hsu, Keerthana Gopalakrishnan, Byron David, Andy Zeng, and Chuyuan Kelly Fu. Do as i can, not as i say: Grounding language in robotic affordances. In 6th Annual Conference on Robot Learning, 2022. URL https://openreview.net/forum?id=bdHkMjBJG_w.

Stephen James, Zicong Ma, David Rovick Arrojo, and Andrew J. Davison. Rlbench: The robot learning benchmark & learning environment. IEEE Robotics and Automation Letters, 5(2):3019–3026, 2020. doi: 10.1109/LRA.2020.2974707.

Justin Johnson, Bharath Hariharan, Laurens van der Maaten, Li Fei-Fei, C. Lawrence Zitnick, and Ross Girshick. Clevr: A diagnostic dataset for compositional language and elementary visual reasoning. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), July 2017.

Amita Kamath, Jack Hessel, and Kai-Wei Chang. What’s ”up” with vision-language models? investigating their struggle with spatial reasoning. In The 2023 Conference on Empirical Methods in Natural Language Processing, 2023. URL https://openreview.net/forum?id=RN5KLywTll.

Ivan Kapelyukh, Yifei Ren, Ignacio Alzugaray, and Edward Johns. Dream2real: Zero-shot 3d object rearrangement with vision-language models. In First Workshop on Vision-Language Models for Navigation and Manipulation at ICRA 2024, 2024. URL https://openreview.net/forum?id=o29sRo5TdE.

Stephen C. Levinson. Space in Language and Cognition: Explorations in Cognitive Diversity. Language Culture and Cognition. Cambridge University Press, 2003.

Xinghang Li, Minghuan Liu, Hanbo Zhang, Cunjun Yu, Jie Xu, Hongtao Wu, Chilam Cheang, Ya Jing, Weinan Zhang, Huaping Liu, Hang Li, and Tao Kong. Vision-language foundation models as effective robot imitators. In The Twelfth International Conference on Learning Representations, 2024a. URL https://openreview.net/forum?id=lFYj0oibGR.

Xuanlin Li, Kyle Hsu, Jiayuan Gu, Oier Mees, Karl Pertsch, Homer Rich Walke, Chuyuan Fu, Ishikaa Lunawat, Isabel Sieh, Sean Kirmani, Sergey Levine, Jiajun Wu, Chelsea Finn, Hao Su, Quan Vuong, and Ted Xiao. Evaluating real-world robot manipulation policies in simulation. In 8th Annual Conference on Robot Learning, 2024b. URL https://openreview.net/forum?id=LZh48DTg71.

Jacky Liang, Wenlong Huang, Fei Xia, Peng Xu, Karol Hausman, Brian Ichter, Pete Florence, and Andy Zeng. Code as policies: Language model programs for embodied control. In 2023 IEEE International Conference on Robotics and Automation (ICRA), pp. 9493–9500, 2023. doi: 10.1109/ICRA48891.2023.10160591.

Xun Liang, Xin Guo, Zhongming Jin, Weihang Pan, Penghui Shang, Deng Cai, Binbin Lin, and Jieping Ye. Enhancing spatial reasoning through visual and textual thinking, 2025. URL https://arxiv.org/abs/2507.20529.

Bo Liu, Yifeng Zhu, Chongkai Gao, Yihao Feng, qiang liu, Yuke Zhu, and Peter Stone. LIBERO: Benchmarking knowledge transfer for lifelong robot learning. In Thirty-seventh Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2023a. URL https://openreview.net/forum?id=xzEtNSuDJk.

Fangyu Liu, Guy Emerson, and Nigel Collier. Visual spatial reasoning. Transactions of the Association for Computational Linguistics, 11:635–651, 2023b. doi: 10.1162/tacl_a_00566. URL https://aclanthology.org/ 2023.tacl-1.37/.

Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. In Thirty-seventh Conference on Neural Information Processing Systems, 2023c. URL https://openreview.net/forum?id= w0H2xGHlkw.

Chenyang Ma, Kai Lu, Ta-Ying Cheng, Niki Trigoni, and Andrew Markham. Spatialpin: Enhancing spatial reasoning capabilities of vision-language models through prompting and interacting 3d priors. In A. Globerson, L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. Tomczak, and C. Zhang (eds.), Advances in Neural Information Processing Systems, volume 37, pp. 68803–68832. Curran Associates, Inc., 2024. URL https://proceedings. neurips.cc/paper_files/paper/2024/file/7f2257d2b291b8d7e712c70b67e09412-Paper-Conference.pdf.

Oier Mees, Lukas Hermann, Erick Rosete-Beas, and Wolfram Burgard. Calvin: A benchmark for languageconditioned policy learning for long-horizon robot manipulation tasks. IEEE Robotics and Automation Letters, 7(3):7327–7334, 2022. doi: 10.1109/LRA.2022.3180108.

Oier Mees, Dibya Ghosh, Karl Pertsch, Kevin Black, Homer Rich Walke, Sudeep Dasari, Joey Hejna, Tobias Kreiman, Charles Xu, Jianlan Luo, You Liang Tan, Dorsa Sadigh, Chelsea Finn, and Sergey Levine. Octo: An open-source generalist robot policy. In First Workshop on Vision-Language Models for Navigation and Manipulation at ICRA 2024, 2024. URL https://openreview.net/forum?id=jGrtIvJBpS.

NVIDIA. Isaac Sim, 2025. URL https://github.com/isaac-sim/IsaacSim.

Zekun Qi, Wenyao Zhang, Yufei Ding, Runpei Dong, Xinqiang Yu, Jingwen Li, Lingyun Xu, Baoyu Li, Xialin He, Guofan Fan, Jiazhao Zhang, Jiawei He, Jiayuan Gu, Xin Jin, Kaisheng Ma, Zhizheng Zhang, He Wang, and Li Yi. Sofar: Language-grounded orientation bridges spatial reasoning and object manipulation, 2025. URL https://arxiv.org/abs/2502.13143.

BAAI RoboBrain-Team, Mingyu Cao, Huajie Tan, Yuheng Ji, Xiansheng Chen, Minglan Lin, Zhiyu Li, Zhou Cao, Pengwei Wang, Enshen Zhou, Yi Han, Yingbo Tang, Xiangqi Xu, Wei Guo, Yaoxu Lyu, Yijie Xu, Jiayu Shi, Mengfei Du, Cheng Chi, Mengdi Zhao, Xiaoshuai Hao, Junkai Zhao, Xiaojie Zhang, Shanyu Rong, Huaihai Lyu, Zhengliang Cai, Yankai Fu, Ning Chen, Bolun Zhang, Lingfeng Zhang, Shuyi Zhang, Dong Liu, Xi Feng, Songjing Wang, Xiaodan Liu, Yance Jiao, Mengsi Lyu, Zhuo Chen, Chenrui He, Yulong Ao, Xue Sun, Zheqi He, Jingshu Zheng, Xi Yang, Donghai Shi, Kunchang Xie, Bochao Zhang, Shaokai Nie, Chunlei Men, Yonghua Lin, Zhongyuan Wang, Tiejun Huang, and Shanghang Zhang. Robobrain 2.0 technical report, 2025. URL https://arxiv.org/abs/2507.02029.

Mohit Shridhar, Jesse Thomason, Daniel Gordon, Yonatan Bisk, Winson Han, Roozbeh Mottaghi, Luke Zettlemoyer, and Dieter Fox. Alfred: A benchmark for interpreting grounded instructions for everyday tasks. In 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 10737–10746, 2020. doi: 10.1109/CVPR42600.2020.01075.

Chan Hee Song, Valts Blukis, Jonathan Tremblay, Stephen Tyree, Yu Su, and Stan Birchfield. Robospatial: Teaching spatial understanding to 2d and 3d vision-language models for robotics. In Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR), pp. 15768–15780, June 2025.

Sanjana Srivastava, Chengshu Li, Michael Lingelbach, Roberto Martín-Martín, Fei Xia, Kent Elliott Vainio, Zheng Lian, Cem Gokmen, Shyamal Buch, Karen Liu, Silvio Savarese, Hyowon Gweon, Jiajun Wu, and Li Fei-Fei. Behavior: Benchmark for everyday household activities in virtual, interactive, and ecological environments. In Aleksandra Faust, David Hsu, and Gerhard Neumann (eds.), Proceedings of the 5th Conference on Robot Learning, volume 164 of Proceedings of Machine Learning Research, pp. 477–490. PMLR, 08–11 Nov 2022. URL https://proceedings.mlr.press/v164/srivastava22a.html.

Balakumar Sundaralingam, Siva Kumar Sastry Hari, Adam Fishman, Caelan Garrett, Karl Van Wyk, Valts Blukis, Alexander Millane, Helen Oleynikova, Ankur Handa, Fabio Ramos, Nathan Ratliff, and Dieter Fox. Curobo: Parallelized collision-free robot motion generation. In 2023 IEEE International Conference on Robotics and Automation (ICRA), pp. 8112–8119, 2023. doi: 10.1109/ICRA48891.2023.10160765.

Andrew Szot, Alex Clegg, Eric Undersander, Erik Wijmans, Yili Zhao, John Turner, Noah Maestre, Mustafa Mukadam, Devendra Chaplot, Oleksandr Maksymets, Aaron Gokaslan, Vladimir Vondrus, Sameer Dharur, Franziska Meier, Wojciech Galuba, Angel Chang, Zsolt Kira, Vladlen Koltun, Jitendra Malik, Manolis Savva, and Dhruv Batra. Habitat 2.0: training home assistants to rearrange their habitat. In Proceedings of the 35th International Conference on Neural Information Processing Systems, NIPS ’21, Red Hook, NY, USA, 2021. Curran Associates Inc. ISBN 9781713845393.

Gemini 2.5 Team, Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, and Others. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities, 2025. URL https://arxiv.org/abs/2507.06261.

Emanuel Todorov, Tom Erez, and Yuval Tassa. Mujoco: A physics engine for model-based control. In 2012 IEEE/RSJ International Conference on Intelligent Robots and Systems, pp. 5026–5033, 2012. doi: 10.1109/IROS.2012.6386109.

Shengbang Tong, Ellis Brown, Penghao Wu, Sanghyun Woo, Manoj Middepogu, Sai Charitha Akula, Jihan Yang, Shusheng Yang, Adithya Iyer, Xichen Pan, Austin Wang, Rob Fergus, Yann LeCun, and Saining Xie. Cambrian-1: A fully open, vision-centric exploration of multimodal llms. In A. Globerson, L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. Tomczak, and C. Zhang (eds.), Advances in Neural Information Processing Systems, volume 37, pp. 87310–87356. Curran Associates, Inc., 2024. URL https://proceedings.neurips.cc/ paper_files/paper/2024/file/9ee3a664ccfeabc0da16ac6f1f1cfe59-Paper-Conference.pdf.

Yuqi Wang, Xinghang Li, Wenxuan Wang, Junbo Zhang, Yingyan Li, Yuntao Chen, Xinlong Wang, and Zhaoxiang Zhang. Unified vision-language-action model, 2025. URL https://arxiv.org/abs/2506.19850.

Fei Xia, Amir R. Zamir, Zhiyang He, Alexander Sax, Jitendra Malik, and Silvio Savarese. Gibson env: Realworld perception for embodied agents. In 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 9068–9079, 2018. doi: 10.1109/CVPR.2018.00945.

Bing Xie, Xiangming Xi, Xinan Zhao, Yuhan Wang, Wei Song, Jianjun Gu, and Shiqiang Zhu. Chatgpt for robotics: A new approach to human-robot interaction and task planning. In ICIRA (5), pp. 365–376, 2023. URL https://doi.org/10.1007/978-981-99-6495-6_31.

Jihan Yang, Shusheng Yang, Anjali W. Gupta, Rilyn Han, Li Fei-Fei, and Saining Xie. Thinking in space: How multimodal large language models see, remember, and recall spaces, 2024. URL https://arxiv.org/abs/2412. 14171.

Rui Yang, Hanyang Chen, Junyu Zhang, Mark Zhao, Cheng Qian, Kangrui Wang, Qineng Wang, Teja Venkat Koripella, Marziyeh Movahedi, Manling Li, Heng Ji, Huan Zhang, and Tong Zhang. EmbodiedBench: Comprehensive benchmarking multi-modal large language models for vision-driven embodied agents. In Aarti Singh, Maryam Fazel, Daniel Hsu, Simon Lacoste-Julien, Felix Berkenkamp, Tegan Maharaj, Kiri Wagstaff, and Jerry Zhu (eds.), Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pp. 70576–70631. PMLR, 13–19 Jul 2025. URL https://proceedings.mlr.press/v267/yang25f.html.

Seonghyeon Ye, Joel Jang, Byeongguk Jeon, Se June Joo, Jianwei Yang, Baolin Peng, Ajay Mandlekar, Reuben Tan, Yu-Wei Chao, Bill Yuchen Lin, Lars Liden, Kimin Lee, Jianfeng Gao, Luke Zettlemoyer, Dieter Fox, and Minjoon Seo. Latent action pretraining from videos. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/forum?id=VYOe2eBQeh.

Tianhe Yu, Deirdre Quillen, Zhanpeng He, Ryan Julian, Karol Hausman, Chelsea Finn, and Sergey Levine. Meta-world: A benchmark and evaluation for multi-task and meta reinforcement learning. In Leslie Pack Kaelbling, Danica Kragic, and Komei Sugiura (eds.), Proceedings of the Conference on Robot Learning, volume 100 of Proceedings of Machine Learning Research, pp. 1094–1100. PMLR, 30 Oct–01 Nov 2020. URL https://proceedings.mlr.press/v100/yu20a.html.

Wentao Yuan, Jiafei Duan, Valts Blukis, Wilbert Pumacay, Ranjay Krishna, Adithyavairavan Murali, Arsalan Mousavian, and Dieter Fox. Robopoint: A vision-language model for spatial affordance prediction in robotics. In 8th Annual Conference on Robot Learning, 2024. URL https://openreview.net/forum?id=GVX6jpZOhU.

Andy Zeng, Pete Florence, Jonathan Tompson, Stefan Welker, Jonathan Chien, Maria Attarian, Travis Armstrong, Ivan Krasin, Dan Duong, Vikas Sindhwani, and Johnny Lee. Transporter networks: Rearranging the visual world for robotic manipulation. In Jens Kober, Fabio Ramos, and Claire Tomlin (eds.), Proceedings of the 2020 Conference on Robot Learning, volume 155 of Proceedings of Machine Learning Research, pp. 726–747. PMLR, 16–18 Nov 2021. URL https://proceedings.mlr.press/v155/zeng21a.html.

Weichen Zhang, Zile Zhou, Zhiheng Zheng, Chen Gao, Jinqiang Cui, Yong Li, Xinlei Chen, and Xiao-Ping Zhang. Open3dvqa: A benchmark for comprehensive spatial reasoning with multimodal large language model in open space, 2025. URL https://arxiv.org/abs/2503.11094.

Baining Zhao, Ziyou Wang, Jianjie Fang, Chen Gao, Fanhang Man, Jinqiang Cui, Xin Wang, Xinlei Chen, Yong Li, and Wenwu Zhu. Embodied-r: Collaborative framework for activating embodied spatial reasoning in foundation models via reinforcement learning, 2025. URL https://arxiv.org/abs/2504.12680.

Peiyuan Zhi, Zhiyuan Zhang, Muzhi Han, Zeyu Zhang, Zhitian Li, Ziyuan Jiao, Baoxiong Jia, and Siyuan Huang. Closed-loop open-vocabulary mobile manipulation with gpt-4v. CoRR, abs/2404.10220, 2024. URL https://doi.org/10.48550/arXiv.2404.10220.

Enshen Zhou, Jingkun An, Cheng Chi, Yi Han, Shanyu Rong, Chi Zhang, Pengwei Wang, Zhongyuan Wang, Tiejun Huang, Lu Sheng, and Shanghang Zhang. Roborefer: Towards spatial referring with reasoning in vision-language models for robotics, 2025. URL https://arxiv.org/abs/2506.04308.

Jinguo Zhu, Weiyun Wang, Zhe Chen, Zhaoyang Liu, Shenglong Ye, Lixin Gu, Hao Tian, Yuchen Duan, Weijie Su, Jie Shao, Zhangwei Gao, Erfei Cui, Xuehui Wang, Yue Cao, Yangzhou Liu, Xingguang Wei, Hongjie Zhang, Haomin Wang, Weiye Xu, Hao Li, Jiahao Wang, Nianchen Deng, Songze Li, Yinan He, Tan Jiang, Jiapeng Luo, Yi Wang, Conghui He, Botian Shi, Xingcheng Zhang, Wenqi Shao, Junjun He, Yingtong Xiong, Wenwen Qu, Peng Sun, Penglong Jiao, Han Lv, Lijun Wu, Kaipeng Zhang, Huipeng Deng, Jiaye Ge, Kai Chen, Limin Wang, Min Dou, Lewei Lu, Xizhou Zhu, Tong Lu, Dahua Lin, Yu Qiao, Jifeng Dai, and Wenhai Wang. Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models, 2025. URL https://arxiv.org/abs/2504.10479.

ABSTRACT

This supplementary material includes (1) details of task definitions (§A.1), including a taxonomy of spatial aspects in Table 10 and curated functional programs in Table 11 and 12, (2) setups of the tabletop scene and the shelf scene (§A.2), (3) a discussion on the sim-to-real relevance (§A.3), (4) evaluation details, such as prompting procedures, essential prompts, and evaluation efficiency (§A.4), and (5) details of ESPIRE assets, including their visualizations and dimensions (§A.5).

A APPENDIX

- A.1 PARTICIPANTS OF A ROBOTICS TASK

A robotics task, in its simplest form, can be defined by an action and a manipulable object (e.g., ‘pick up the book’). We employ two primitive actions, ‘pick’ and ‘place’, to initiate robotics tasks. Keeping the action space simple helps isolate spatial reasoning behaviors, allowing for a focus on their analysis.

To highlight the facets of spatial reasoning and support systematic task design and experimental analysis, we categorize key spatial aspects (S), reference frames (F), and reference objects (O) that characterize spatial reasoning and combine them to define task specifications (see Section 4.1).

- • Reference frames. Reference frames refer to coordinate systems essential for describing one object in relation to another. They can be made explicit via linguistic specifications, but are usually implicitly conveyed within the context. Following Levinson (2003), we consider three types of reference frames.

- – Relative frames are viewer-centered; for example, ‘behind the mirror’ may refer to the space further from the viewer, from the viewer’s perspective toward the mirror.
- – Intrinsic frames are object-centered; for example, ‘behind the mirror’ may indicate the space opposite to the mirror’s facing direction, independent of the viewer.
- – Absolute frames are defined with respect to fixed global coordinates, such as elevation and altitude (useful for describing below and above) and cardinal directions (e.g., north and east), but are only used in a few indoor scenarios.

- • Objects. ESPIRE contains two primary object types: manipulable and reference objects.

- – Manipulable objects are instantiated as cuboid-shaped books (see Table 18). The regular geometries make it easier to verify their final states, facilitating automated evaluation. Moreover, they yield a relatively high likelihood of generating valid grasping/placement poses, without relying on external tools for pose proposal, yet remain sufficiently challenging for 6-DoF tasks.
- – Reference objects participate in describing an object in relation to another. In cases where a reference frame is not explicitly specified, the intrinsic frame of the referenced object may be used. Thus, we divide reference objects into intrinsic-oriented objects that have a clear front face (e.g., a chair or mirror) and non-oriented objects that do not (e.g., a jar or ball). To support fine-grained analysis of spatial reasoning, such as distinguishing units of measure in distance estimation (meter vs. centimeter), we further divide reference objects into near and distant categories (see Table 16 and 17). Near objects appear on the shelf or tabletop, whereas distant objects are located outside these areas.

- • Spatial aspects. We group spatial aspects into four broad classes: attributes, distances, relationships, and orientations (see an overview in Table 10). Whenever applicable, we consider both coarse- and fine-grained expressions, such as relative distance and precise distance:

- – Attributes primarily refer to intrinsic size attributes (i.e., dimensions) of an object, such as height, length, width, volume, and diameter/radius. They may be implicitly used to check space fitness and describe object volume, e.g., a large/small book.
- – Distances describe the proximity between objects. Apart from relative distance descriptions like nearest, farthest, and second farthest, we include precise distance descriptions using different units of measure, e.g., within 1 meter of the mirror and 20 centimeters away from the jar.’

[Figure 41]

[Figure 42]

#### 5.0

0.8

###### 0.7

1.5

4.0

1.4

Shelf

1.2

1.0

Tabletop

0.6

[Figure 43]

0.2

0.4

1.0

###### 0.4

0.7

1.2

0.2

1.0

Figure 3: Layouts of the tabletop and shelf scenes within ESPIRE (best viewed in color). The light red region denotes the camera viewpoint sampling area, the light green region indicates where the robot end effector may appear, and the light blue region denotes where distant reference objects are placed. All labeled dimensions are in meters.

- – Relationships primarily describes positional relations, i.e., how one object is positioned relative to another. They can be expressed in diverse ways in natural language, but we consider only the most commonly used basic forms like left, right, in front of, behind, below, and above, and their comparative and superlative forms like leftmost, rightmost, and second leftmost.
- – Orientations cover directional expressions, including coarse-grained state descriptions (e.g., upright and at a tilt) and fine-grained clock positions (e.g., to your 6 o’clock) and degrees of a tilt (e.g., at a 45-degree tilt).

We note that our definition of the task specification (C = (S,F,O)) primarily disentangles the complexity of spatial reasoning over ‘Relationships’ and ‘Orientations’, as they rely on a frame of reference, but ‘Attributes’ and ‘Distances’ do not. Nonetheless, we use this definition across all four spatial aspects to keep consistent.

- A.2 SIMULATED ENVIRONMENT

We focus on tasks that involve picking up an object from the table and placing it on the shelf (see Figure 3). Though the reverse direction—picking up an object from the shelf and placing it on the table—is also feasible and would increase task diversity, we consider only the former because it alone suffices to cover a diverse range of spatial reasoning scenarios.

Tabletop tasks. The table scene is initialized with manipulable books, support ornaments, and reference ornaments. For books, we consider three different sizes (i.e., small, medium, and large), and three different initial poses (i.e., standing upright, lying flat, and at a tilt). We use a small set of support ornaments to create flat and tilting poses while ensuring that the books are pickable via 6-DoF pose prediction. Near references are small-sized, appearing on the table (e.g., a picture frame or ceramic jar), whereas distant references are large-sized, located on the floor behind the table (e.g., a floor lamp or cheval mirror). Note that the reference ornaments are carefully selected to cover intrinsic-oriented and non-oriented categories, and the robot itself is an intrinsic-oriented reference, always facing the front of the table. We initialize the tabletop with two random near objects and one random distant object. The degree of clutter in the scene is controlled by varying the number of books on the table, while the overall complexity is driven by instructions and environmental factors like varying poses, lights, and textures.

The global camera looks at a random point on the front edge of the table. We randomly sample its elevation 0.5–1m above the table surface. The elevation of the end-effector is randomly sampled from 0.3–0.5m below the global camera, with pitch, yaw, roll randomly sampled from [−22.5◦,22.5◦], [−22.5◦,22.5◦], and [−45◦,45◦], respectively.

Shelf tasks. The shelf scene contains the same types of objects as in the tabletop scene. Analogously to the tabletop scenes, where books are initialized in random grasping poses, we initialize the shelf with random support ornaments to let books lean against, creating various placement poses. Compared with the tabletop scene, the shelf scene supports reasoning of two additional spatial relationships: above and below. We locate distant references on the floor, either to the left or to the right of the shelf. The robot always faces the front of the shelf. We control the complexity of shelf tasks by varying shelf layouts, including horizontal panels, grids of slots, and their combinations.

The global camera always faces the shelf center. We randomly sample its elevation 1.2–1.5m above the ground. The elevation of the end-effector is randomly sampled from 0.3–0.5m below the global camera, with pitch, yaw, roll randomly sampled from [−22.5◦,22.5◦], [−22.5◦,22.5◦], and [−120◦,−60◦] ∪ [60◦,120◦], respectively.

Definitions of spatial relationships and orientations. In natural language, spatial relationships can exhibit ambiguity due to the reliance on reference frames and contexts. To address this issue, we use a unified definition to assign them unambiguous geometric interpretations. Specifically, under a given reference frame, we use its forward axis to represent the front-facing direction, then left and right are defined relative to it. The definition of behind is, however, more involved, as it depends on the reference frame. Suppose the description ‘O1 is behind O2’, when the reference frame is independent of O2, it is interpreted as: O1 is further than O2 along the front-facing direction; when the reference frame is attached to O2, the meaning changes to: O1 is further along the opposite of the front-facing direction.

We account for two fine-grained types of orientation: direction and tilt. Directions are represented using clock positions that provide a granular description relative to a specific reference frame. In this setup, the forward axis is assigned to 12 o’clock, with all other positions mapped relative to this heading. To describe precise tilts, we measure the tilt angle in degrees and define it as the deviation between the global up-axis and the upright axis of the object.

Definitions of ‘above’ and ‘below’. The global up-axis corresponds to the up-direction of an absolute reference frame, defined as the surface normal of the floor in our simulation environment. We also rely on this global up-axis to define spatial relationships like above and below. Specifically, ‘O1 is above O2’ indicates that O1 lies further along the global up-axis; equivalently, ‘O2 is below O1’. Following the standard convention, we define the tilt angle as the angle between the up axis and the surface normal of an object.

Mitigation of ambiguity. For objects involved in tabletop tasks, we randomly initialize their locations while ensuring that they are spaced at least 5cm apart. Note that due to physical rendering constraints, the final spacing may be smaller than 5cm. We require that at least 20% of the pixels of each object are visible in the global view. For shelf tasks, we require that at least 50% of the pixels of the book in hand are visible. When a target satisfies multiple constraints (e.g., a book can be behind the picture while also being to its left), we select the most salient one for task generation.

Balanced task sampling. To ensure that ESPIRE tasks are approximately uniformly distributed across task families T and difficulty levels L = {easy,medium,hard}, we propose a balanced task sampling strategy (see Algorithm 1). Specifically, we maintain a counter Ct,l for each combination of task family t ∈ T and difficulty level l ∈ L. We also record the number of times each scene s ∈ S has been attempted so far, denoted by As. These counters are used to dynamically adjust the sampling weights (lines 15 and 21).

During task generation, we first select a task family with preference for underrepresented families (lines 12–15). Given the selected task family, we collect all scenes that yield a non-empty answer set and randomly sample one, favoring underrepresented difficulty levels while penalizing scenes that have been repeatedly attempted (lines 16–19). We repeat this process until the desired number of tasks has been generated.

- Algorithm 1 Balanced Task Sampling

- 1: Input: Task families T , scenes S, difficulty levels L = {easy, medium, hard}, the number of tasks per family Nt, the number of tasks in total Nall.
- 2: Initialize task counts {Ct,l}t∈T ,l∈L ← 0
- 3: Initialize scene attempts {As}s∈S ← 0
- 4: Initialize the task set Q ← ∅
- 5: N ← 0
- 6: while ELIGIBLEFAMILY(T , S) ̸= ∅ do
- 7: if Nall is defined and N ≥ Nall then
- 8: break
- 9: end if
- 10: Tsub ← {t ∈ T | l Ct,l < Nt, GETCOMPATIBLESCENES(t, S) ̸= ∅}
- 11: if Tsub = ∅ then
- 12: break
- 13: end if
- 14: for all t ∈ Tsub do
- 15: wt ←

1 l Ct,l + 1

▷ Under-sampled task families

- 16: end for
- 17: t⋆ ← WEIGHTEDSAMPLING(Tsub, {wt})
- 18: St⋆ ← GETCOMPATIBLESCENES(t⋆, S)
- 19: for all s ∈ St⋆ do
- 20: l ← GETDIFFICULTYLEVEL(t⋆, s)
- 21: ws ←

1 t′ Ct′,l + 1 ·

1 (As + 1)2

▷ Under-sampled scenes and difficulty levels

- 22: end for
- 23: s⋆ ← WEIGHTEDSAMPLING(St⋆, {ws})
- 24: q ← GENERATEANSWERSET(t⋆, s⋆)
- 25: As⋆ ← As⋆ + 1
- 26: if |q| > 1 then ▷ Retain only non-trivial tasks
- 27: l⋆ ← GETDIFFICULTYLEVEL(s⋆, t⋆)
- 28: Q ← Q ∪ {(t⋆, s⋆)}
- 29: Ct⋆,l⋆ ← Ct⋆,l⋆ + 1
- 30: N ← N + 1
- 31: end if
- 32: end while
- 33: return Q

- A.3 SIM-TO-REAL RELEVANCE

To confirm ESPIRE serves as a reliable proxy for embodied spatial reasoning, we establish the benchmark’s validity through the following two lens:

Performance alignment. We evaluated Qwen3-VL (8B/30B/235B), RoboBrain2.0-7B and Gemini2.5-Pro on ESPIRE and the pointing tasks of the natural-image benchmark RefSpatial (Zhou et al., 2025). We then compute Spearman’s rank correlation between the model performance rankings on the two benchmarks. The resulting correlation is 96.4% (with p = 0.00498), indicating strong alignment between the two evaluations and suggesting that ESPIRE serves as a high-fidelity proxy for real-world embodied spatial reasoning.

Human study. We conducted a study with five humans to assess environment realism and model alignment (see Section 5.5). First, we observe an average 94.9 ± 3.4% human success rate across all tasks, suggesting that the simulated scenarios are readily interpretable and solvable by humans.

We further analyze agreement on the ground-truth reference frame across three reference categories: near oriented objects, distant oriented objects, and the table. Specifically, we measure the proportion of examples with unanimous agreement among the five annotators. For examples involving distant oriented references and the table, humans agree on the reference frame in more than 97% of cases, suggesting that the intended frame is clearly interpretable in these settings. However, only 31.03% of the examples involving near oriented references yield unanimous agreement. This lower agreement

- Algorithm 2 Localization Procedures (a) Localization w/o Reflection

- 1: Input: Task instruction T, Scene observation O, Maximum trials N
- 2: for i = 1 to N do
- 3: P ← PREDICT(T, O)
- 4: R ← EVALUATE(P)
- 5: if R.success then
- 6: break ▷ Stop if successful
- 7: end if
- 8: end for

(b) Localization w/ Reflection

- 1: Input: Task instruction T, Scene observation O, Maximum trials N
- 2: F, R0 ← None
- 3: for i = 1 to N do
- 4: P ← PREDICT(T, O, F, R0)
- 5: R1 ← EVALUATE(P)
- 6: if R.success then
- 7: break
- 8: end if
- 9: F ← REFLECT(T, O, R)
- 10: R0 ← R1
- 11: end for

- Algorithm 3 Execution Procedures

(a) Execution w/o Reflection

(b) Execution w/ Reflection

- 1: Input: Task configuration T, Scene observation O, Maximum trials N
- 2: for i = 1 to N do
- 3: P ← PREDICT(T, O)
- 4: R ← EVALUATE(P)
- 5: if R.success then
- 6: break ▷ Stop if task is done
- 7: end if
- 8: O ← GETOBSERVATION()
- 9: end for

- 1: Input: Task configuration T, Initial observation O, Maximum trials N
- 2: F, O0, R0 ← None
- 3: for i = 1 to N do
- 4: P ← PREDICT(T, O, F, O0, R0)
- 5: R1 ← EVALUATE(P)
- 6: if R1.success then
- 7: break
- 8: end if
- 9: O0 ← O
- 10: R0 ← R1
- 11: O ← GETOBSERVATION()
- 12: F ← REFLECT(T, O, O0, R0)
- 13: end for

likely reflects the inherent ambiguity of reasoning with nearby oriented objects, as discussed in Section 5.5.

- A.4 EVALUATION

Task status checking. After execution, we obtain task status (e.g., failure or success) by checking if the final environment state satisfies the constraints specified in the instruction. (1) Distance is measured as the minimum distance between the boundaries of the target and the reference object (which can be a 3D point). The final distance within ±3 cm of the expected distance is considered correct in the evaluation. (2) Orientation and Relationship are determined by checking if the center of the target lies in the target area defined by a reference frame. The final tilt angle within ±10◦ of the expected angle is considered correct in the evaluation.

Algorithms. We illustrate the evaluation procedures used for localization in Algorithm 2a (w/o reflection) and Algorithm 2b (w/ reflection). Reflection is performed following a localization failure (line 9 of Algorithm 2b). The generated reflection tokens are added to the inputs to the next iteration (line 4 of Algorithm 2b). The evaluation procedures for execution are illustrated in Algorithm 3a (w/o reflection) and Algorithm 3b (w/ reflection). They are similar to those used for localization except that the reflection for execution relies on an additional view of the failure state (line 9–12 of Algorithm 3b).

Prompts. We provide our customized prompts for pick tasks with Qwen3-VL (Bai et al., 2025). Figures 4a, 4b, 5a, and 5b show the prompts used for localization, execution, localization w/ reflection, and rotation, respectively. They are different across VLMs primarily in the output format, e.g., Gemini2.5-Pro (Team et al., 2025) outputs point coordinates in [y,x] while others in [x,y]. The differences in the prompts for pick and place tasks arise primarily in the task descriptions. For

Table 9: Breakdowns of evaluation time in seconds (s).

Pick Place localization (s) execution (s) # update success (%) localization (s) execution (s) # update success (%)

Models

w/o reflection

Gemini2.5-Pro 22.37 40.53 6.86 34.06 33.00 41.15 9.01 15.70 InternVL3-78B 14.44 32.21 6.92 17.26 19.06 34.04 7.59 9.67 RoboBrain2.0-7B 14.17 27.56 9.07 10.87 19.08 30.83 9.36 8.64 Qwen3-VL-8B 15.44 34.87 6.73 29.32 19.48 35.86 7.83 12.41 Qwen3-VL-30B-A3B 12.50 30.61 6.86 32.15 17.38 32.59 7.39 20.00 Qwen3-VL-235B-A22B 16.08 34.49 7.38 26.76 20.40 37.55 7.64 19.34

w/ reflection

Qwen3-VL-8B 23.94 44.88 8.58 15.07 31.22 47.58 9.31 6.67 Qwen3-VL-30B-A3B 21.51 44.70 8.37 17.08 27.99 49.29 8.13 13.80 Qwen3-VL-235B-A22B 33.23 66.83 8.12 23.20 44.03 68.58 8.53 15.40

example, a localization instruction for place tasks could be: ‘Given a scene image and a textual description specifying the placement conditions for a book currently held by a robot gripper, you are required to determine the exact placement location in the image.’

Evaluation Time. We break down the evaluation time along the evaluation stage (see Table 9). The results are averaged across all tasks and attempts. We record the time taken until a successful attempt is achieved. During execution, the environment updates after each move and in each observation query, so we also report the average number of environment updates over successful tasks. Compared to the model inference time in localization and execution, the environment update is quite quick, i.e., it takes an average of 11.65 seconds per update. Models with reflection enabled generally take longer because they require additional API calls. A higher number of environment updates indicates more execution attempts. For example, RoboBrain2.0-7B (RoboBrain-Team et al., 2025) not only requires the largest number of environment updates but also achieves the lowest success rate, suggesting its weaker capability in execution.

Running Examples. We provide running examples with Qwen3-VL-235B-A22B (Bai et al., 2025). Figure 6 shows an excerpt of the localization (w/ reflection) logs of a pick task, and Figure 8 shows an example run on a place task without reflection.

Figure 9 shows an excerpt of the execution (w/o reflection) logs of a pick task. Note that, in this example, the model also needs to predict rotations for the pitch axis. Figure 10 presents an excerpt of the execution (w/ reflection) logs of a place task.

In this task, the model must predict a goal position for the end-effector. Interestingly, the model fails in the first attempt but succeeds by leveraging reflection in the second attempt.

- A.5 ASSET VISUALIZATION

We visualize primary assets of ESPIRE, including near reference objects in Table 16, distant reference objects in Table 17, tables in Table 13, shelf layouts in Table 14, shelf textures in Table 15, manipulable books in Table 18, and support ornaments in Table 19.

You are an expert Visual-Language Assistant specialized in embodied pointing tasks. Note that any occurrence of “You” denotes the first-person camera viewpoint of the RGB image (i.e., the perceptual perspective). It does NOT refer to any object, robot, or robotic end-effector visible in the image.

Task: Identify the object in an RGB image according to a detailed textual description and provide precise coordinates for the object that match the input conditions.

Inputs: 1. A text description specifying the book that needs to be located, including all required attributes and conditions. 2. An RGB image representing the scene, captured from your perceptual viewpoint.

Output: Locate the target object with a point, report its point coordinate in JSON format like this: “point_2d”: [x, y]

(a) Localization in pick tasks.

You are an expert Visual-Language Assistant specialized for embodied robot-arm manipulation.

Task: Determine the target position of the robotic arm’s end-effector to grasp a specific book, where the book is marked with a colored bounding box. Choose a grasp point that is easily reachable and suitable for the robot’s end-effector, avoiding edges or positions that are obstructed or unstable. The grasp point should allow the robot to securely pick up the book without collisions or slippage.

Note: If the predicted target point lies on the target book, the robot will attempt to directly grasp the book at that location. If the predicted point lies on another object or a free-space region, the robot will move its end-effector to the indicated position.

Inputs: 1. An RGB image representing the scene, where the target book is marked with a colored bounding box. 2. A textual instruction that specifies how the robot should grasp the book.

Output: Locate the target position with a point, report its point coordinate in JSON format like this: “point_2d”: [x, y]

(b) Execution in pick tasks.

Figure 4: Example prompts with Qwen3-VL (continued).

You are an expert Visual-Language Assistant specialized in reflective analysis of embodied pointing tasks.

Task: Analyze the previous prediction error in the image grounding task. The incorrect predicted point has been marked with a red circle on the RGB image. Reflect on why the model’s prediction failed and explain what visual or textual cues were misunderstood or ignored.

Inputs: 1. The original textual description specifying the target object and its required attributes or conditions. 2. The same RGB image used in the grounding task, now containing the red circle marking the incorrect prediction.

Output: Provide a detailed textual reflection explaining: - The likely cause of the failure (e.g., visual confusion, incorrect attribute matching, spatial misinterpretation, etc.). - The correct reasoning process that should have been applied. - Suggestions for how to improve the next prediction.

(a) Localization w/ reflection in pick tasks.

You are an expert Visual-Language Assistant specialized for embodied robot-arm manipulation and precise grasp orientation control.

Task: Given an RGB image with a colored bounding box marking the target book and a red dot indicating the moving target position, together with a textual instruction specifying the book attributes and required rotation axes (pitch, yaw, roll), predict the end-effector’s **rotation offsets** relative to its current orientation. All rotations are defined in the **target book’s local coordinate frame**.

Inputs: 1. RGB image: - Target book marked by a colored bounding box. - Moving target position marked by a red dot. 2. Text instruction describing the target book and specifying which rotation axes to output.

Reference Frame: Rotations follow the book’s intrinsic axes (forward–backward, left–right, up–down): - **Pitch:** tilt up/down. - **Yaw:** swing left/right. - **Roll:** roll clockwise/counterclockwise. Positive/negative follow standard right-hand conventions.

Output: Return the predicted rotation offsets (degrees) in JSON: “‘json “rotation”: [pitch, yaw, roll]

(b) Rotation prediction in pick tasks, summarized by GPT-5.1 for the demonstration purpose.

Figure 5: Example prompts with Qwen3-VL.

ENV: {task_instruction: from the items on the table, find a book more than 120cm away from you, and fetch it. category: pick}

[Figure 44]

- VLM: {system_prompt: [see Figure 4a], user: from the items on the table, find a book more than 120cm away from you, and fetch it. assistant: {point_2d: [757, 448], label: book more than 120cm away from you}}

[Figure 45]

ENV: {moved: false, success: false}

[Figure 46]

- VLM: {system_prompt: [see Figure 5a], user: The previous visual grounding attempt for the [task_instruction] was incorrect. The red circles in the image mark the wrong grounding regions. Please carefully analyze the image and the instruction again, understand why the previous prediction was wrong, and describe what should be corrected in the next attempt. assistant: The previous prediction was incorrect because it failed to properly interpret the spatial constraint in the instruction... Why the Previous Prediction Was Wrong: 1. Misinterpretation of Spatial Distance 2. Focus on Object Type Only... What Should Be Corrected in the Next Attempt:

[Figure 47]

[Figure 48]

1. Incorporate Spatial Reasoning 2. Re-evaluate All Books on the Table...}

VLM: {system_prompt: [see Figure 4b], user: Here is a reflection on the previous failed attempt and some suggestions: [reflection]. Now, complete this task: from the items..., assistant: {point_2d: [745, 400]}}

[Figure 49]

ENV: {moved: false, success: false}

[Figure 50]

Figure 6: Pick localization example with reflection, where the prompt simplified for better demonstration.

VLM: {system_prompt: [see Figure 5a], user: The previous visual grounding attempt for the [task_instruction] was incorrect. The red circles in the image mark the wrong grounding regions. Please carefully analyze the image and the instruction again, understand why the previous prediction was wrong, and describe what should be corrected in the next attempt. assistant: The previous prediction was incorrect because it focused on a region on the wall... Why the Error Occurred: - Misinterpretation of “items on the table” - Failure to filter by object class... What Should Be Corrected in the Next Attempt: 1. Identify all books on the table 2. Estimate distance from the agent (robot arm)...}

[Figure 51]

[Figure 52]

VLM: {system_prompt: [see Figure 4b], user: Here is a reflection on the previous failed attempt and some suggestions: [reflection]. Now, complete this task: from the items..., assistant: {point_2d: [835, 555]}}

[Figure 53]

ENV: {moved: false, success: true}

[Figure 54]

Figure 7: Pick localization example (subsequent attempts and successful execution).

ENV: {task_instruction: position the book in row 1, column 3 of

[Figure 55]

the shelf. category: place}

VLM: {system_prompt: [see Figure 4a], user: position the book in row 1, column 3 of the shelf. assistant: {point_2d: [640, 224]}}

[Figure 56]

ENV: {moved: false, success: true}

[Figure 57]

- Figure 8: Place localization example without reflection, where the prompt simplified for better demonstration.

ENV: {task_instruction: from the items on the table, find a book more than 120cm away from you, and fetch it. category: pick}

[Figure 58]

VLM: {system_prompt: [see Figure 4b], user: reach for the book that needs to be grabbed; this book is marked by a bounding box of the red color. assistant: {point_2d: [825, 541], label: target book}} {system_prompt: [see Figure 5b], user: [task_instruction] + The rotation delta(s) you should predict: pitch, assistant: {rotation: [5.0, null, null]}}

[Figure 59]

ENV: {moved: true, success: true}

[Figure 60]

- Figure 9: Pick execution example without reflection, where the prompt simplified for better demonstration.

ENV: {task_instruction: position the book in row 1, column 3 of

[Figure 61]

the shelf. category: place}

VLM: {system_prompt: [see Figure 4b], user: Reach for the location where the book in your hand needs to be placed. The book in your hand should be placed in the vertical manner, and the placement area is marked by a bounding box in the cyan color. assistant: {point_2d: [587, 194]}}

[Figure 62]

ENV: {moved: true, success: false}

[Figure 63]

VLM: {system_prompt: [see Figure 5a], user: The previous visual moving attempt for the [task_instruction] was incorrect. The red circles in the image mark the wrong moving regions. Please carefully analyze the image and the instruction again, understand why the previous prediction was wrong, and describe what should be corrected in the next attempt. assistant: The failure in the previous attempt stems from a misinterpretation of the spatial relationship between the cyan bounding box and the robot’s reachable workspace... Likely Cause of Failure: Target Point Misplacement... Kinematic limits... Suggestions for Improving the Next Prediction: Focus on the shelf surface, not the back wall... Prioritize reachable points...}

[Figure 64]

[Figure 65]

VLM: {system_prompt: [see Figure 4b], user: Here is a reflection on the previous failed attempt and some suggestions: [reflection]. Now, complete this task: Reach for the location ..., assistant: {point_2d: [684, 418]}}

[Figure 66]

ENV: {moved: true, success: true}

[Figure 67]

- Figure 10: Place execution example with reflection, where the prompt simplified for better demonstration.

- Table 10: Spatial aspects of varying granularities. nil indicates no input parameters are required. Note that left, right, front, behind, above, and below are reloaded as directional relations in ‘Orientation’ (cf. positional relations in ‘Relationship’).

Spatial Aspect

Granularity

Type Input Example Instruction

Small nil take a small book from the table Medium nil take a medium-sized book Large nil take a large book Empty nil place the book in an empty slot NonEmpty nil place the book in a partly occupied slot Emptiest nil place the book in the emptiest slot

Coarse

Attribute

Height (#,) take a book around 20 centimeters high Width (#,) place the book in a slot around 45 centimeters wide Index1D (#,) place the book at row 2 of the shelf Index2D (#,#) place the book at row 2, column 3 of the shelf

Fine

Closest nil take a book among the books closest to you Farthest nil take a book among the books farthest from you LessThan (#,) place the book in a slot within 1.5 meters of you MoreThan (#,) take a book more than 1.5 meters away from you

Coarse

Distance

RankClosest (#,) take a book among the books second closest to you RankFarthest (#,) take a book among the books second farthest from you EqualTo (#,) take a book about 1.5 meters away from you Range (#,#) take a book 1.5 to 2 meters away from you

Fine

Left nil take a book on the left of the table Right nil place the book in a slot on the left of the shelf Front nil take a book at the front of the table Behind nil take a book at the back of the table Upper nil place the book in a slot in the upper part of the shelf Lower nil place the book in a slot in the lower part of the shelf LeftMost nil take the leftmost book from the table RightMost nil place the the book in a leftmost slot on the shelf

Relationship

Coarse

RankLeftMost (#,) take the second leftmost book on the table RankRightMost (#,) place the book in a second rightmost slot on the shelf Between (#,#) place the book between the alarm clock and the succulents

Fine

Flat nil take a flat-lying book from the table Vertical nil place the book upright on the shelf Tilted nil place the book at a tilt on the shelf Left nil take a book to your left Right nil place the book in a slot to your right Front nil place the book in front of the teddy bear Behind nil take a book behind the picture frame Above nil place the book in a slot above the picture frame Below nil place the book in a slot below the picture frame

Coarse

Orientation

DirectLeft nil place the book immediately to the left of the alarm clock DirectRight nil place the book immediately to the right of the succulents DirectAbove nil place the book in a slot directly above the alarm clock DirectBelow nil place the book in a slot directly below the picture frame ClockPosition (#,) place the book in a slot to your 6 o’clock TiltDegree (#,) place the book at a tilt angle of about 30 degrees

Fine

- Table 11: Example instruction families of pick tasks. The outermost pick(·) is discarded for simplicity. unique(·) ensures a unique item from the input set. TABLE returns all items in the tabletop scene.

S F O R I Example Program

Small Large filterAttr$R(filterBook(TABLE)) Medium

Attribute

Height

float filterAttr$R(I, filterBook(TABLE)) Width

RankClosest

int filterDist$R(I, filterBook(TABLE), O)

RankFarthest LessThan MoreThan list filterDist$R(I, filterBook(TABLE), O) EqualTo Range

viewer distant obj. near obj.

Distance

Left Right filterRel$R(filterBook(TABLE), O) Front Behind RankLeftMost

intrinsic

table

Relationship

int filterRel$R(I, filterBook(TABLE), O)

RankRightMost Between list

filterRel$R(filterBook(TABLE), filter(I1, TABLE), filter(I2, TABLE))

Left

filterRel$R(filterBook(TABLE), O)

relative

Right RankLeftMost

viewer

int filterRel$R(I, filterBook(TABLE), O) RankRightMost

Left Right filterOri$R(filterBook(TABLE), O) Front Behind RankLeftMost

intrinsic

viewer oriented

int filterOri$R(I, filterBook(TABLE), O)

RankRightMost Flat Vertical filterOri$R(filterBook(TABLE)) Tilted ClockPosition int filterOri$R(I, filterBook(TABLE), O) TiltDegree float filterOri$R(I, filterBook(TABLE), O)

Orientation

Left Right filterOri$R(filterBook(TABLE), O) Front Behind RankLeftMost

relative

viewer non-oriented

int filterOri$R(I, filterBook(TABLE), O)

RankRightMost ClockPosition int filterOri$R(I, filterBook(TABLE), O) TiltDegree float filterOri$R(I, filterBook(TABLE), O)

- Table 12: Example instruction families of place tasks. The outermost place(·) is discarded for simplicity. unique(·) ensures a unique item from the input set. SHELF returns all shelf-scene items.

S F O R I Example Program

Small Large filterAttr$R(filterSlot(SHELF)) Medium

Attribute

Height

float filterAttr$R(I, filterSlot(SHELF)) Width

RankClosest

int filterDist$R(I, filterSlot(SHELF), O)

RankFarthest LessThan

viewer distant obj. near obj.

Distance

filterDist$R(I, filterSlot(SHELF), O) filterDist$R(I, filterSpace(SHELF), O)

MoreThan EqualTo Range

list

Left

filterRel$R(filterSlot(SHELF), O) filterRel$R(filterSpace(SHELF), O)

Right Upper Lower

intrinsic

shelf

Relationship

RankLeftMost

int filterRel$R(I, filterSlot(SHELF), O)

RankRightMost Between list

filterRel$R(filterSpace(SHELF), filter(I1, SHELF), filter(I2, SHELF))

Left filterRel$R(filterSlot(SHELF), O) Right filterRel$R(filterSpace(SHELF), O)

relative

viewer

RankLeftMost

int filterRel$R(I, filterSlot(SHELF), O) RankRightMost

Left

filterOri$R(filterSlot(SHELF), O) filterOri$R(filterSpace(SHELF), O)

Right Front Behind

intrinsic

viewer oriented

RankLeftMost

int filterOri$R(I, filterSlot(SHELF), O)

RankRightMost ClockPosition int filterOri$R(I, filterSpace(SHELF), O)

Orientation

Left

filterOri$R(filterSlot(SHELF), O) filterOri$R(filterSpace(SHELF), O)

Right Front Behind

relative

viewer non-oriented

RankLeftMost

int filterOri$R(I, filterSlot(SHELF), O)

RankRightMost ClockPosition int filterOri$R(I, filterSpace(SHELF), O)

Flat

placeOri$R(I, unique(filterSlot(SHELF))) placeOri$R(I, unique(filterSpace(SHELF)))

Vertical Tilted TiltDegree

float

absolute

distant obj. near obj.

Above filterOri$R(filterSlot(SHELF), O) Below filterOri$R(filterSpace(SHELF), O)

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

Table 1 Table 2 Table 3 Table 4

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

Table 5 Table 6 Table 7 Table 8

Table 13: Tables with different colors and textures.

[Figure 76]

[Figure 77]

[Figure 78]

Shelf 1 Shelf 2 Shelf 3

[Figure 79]

[Figure 80]

[Figure 81]

Shelf 4 Shelf 5 Shelf 6

- Table 14: Shelf with different layouts.

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

- Table 15: Shelf with different textures.

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

Alarm Clock Armento Rider Bicycle Sculpture Picture Frame

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

Teddy Bear Newton’s Cradle Geosphere Sculpture Pillar Bookend

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

Rubik’s Cube Succulents Ceramic Jar Pagoda Statue

- Table 16: Near reference objects (w/ and w/o an intrinsic frame of reference) appear on the table or shelf.

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

Cheval Mirror Lady with an Ermine Mona Lisa Adoration of the Magi

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

Floor lamp 1 Floor lamp 2 Magnolia sieboldii Philadelphus shrub

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

Juniperus communis Copper scale Decorative disk Marble bust

- Table 17: Distant reference objects (w/ and w/o an intrinsic frame of reference) appear behind the table or besides the shelf.

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

- Table 18: Manipulable books. They will be auto-scaled to match three pre-defined sizes: small, medium, and large.

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

Table 19: Different bookends are used to help create tilting poses of books.

Table 20: Attributes of ESPIRE assets. Oriented assets have an intrinsic frame while non-oriented assets do not.

Name Type Oriented L (cm) W (cm) H (cm) Alarm clock near ✓ 7 13 17 Armento Rider near ✓ 24 7 24 Bicycle sculpture near ✓ 21 8 18 Picture frame near ✓ 13 22 18 Teddy bear near ✓ 20 23 25 Newton’s cradle near ✗ 10 15 14 Geosphere sculpture near ✗ 15 15 15 Pillar bookend near ✗ 7 16 13 Rubik’s cube near ✗ 6 6 6 Succulents near ✗ 17 15 29 Ceramic jar near ✗ 6 6 8 Pagoda statue near ✗ 13 14 21 Cheval mirror distant ✓ 40 42 43 Lady with an Ermine distant ✓ 62 59 52 Mona Lisa distant ✓ 62 54 52 Adoration of the Magi distant ✓ 62 79 52

- Floor lamp 1 distant ✗ 48 48 59
- Floor lamp 2 distant ✗ 51 51 60 Magnolia sieboldii distant ✗ 61 65 51 Philadelphus shrub distant ✗ 63 61 06 Juniperus communis distant ✗ 46 45 34 Copper scale distant ✗ 52 61 46 Decorative disk distant ✓ 52 52 43 Marble bust distant ✓ 52 52 53 Table 1 − 8 table ✓ 60 140 70

- Shelf 1 shelf ✓ 45 149 215
- Shelf 2 shelf ✓ 45 176 190
- Shelf 3 shelf ✓ 45 176 190
- Shelf 4 shelf ✓ 45 171 190
- Shelf 5 shelf ✓ 45 149 190
- Shelf 6 shelf ✓ 45 164 187 Book-small book ✗ 17.5 – 18.8 10.8 – 13.0 1.5 – 1.8 Book-medium book ✗ 21.6 – 25.0 14.0 – 17.6 2.0 – 2.5 Book-large book ✗ 25.4 – 30.5 20.3 – 24.1 3.7 – 4.0

