## MolmoSpaces

###### A Large-Scale Open Ecosystem for Robot Navigation and Manipulation

Yejin Kim♥1∗ Wilbert Pumacay♥1∗ Omar Rayyan♥3∗ Max Argus♥1∗ Winson Han♥1 Eli VanderBilt♥1 Jordi Salvador♥1 Abhay Deshpande♥1 Rose Hendrix♥1 Snehal Jauhri♥5 Shuo Liu♥2 Nur Muhammad Mahi Shafiullah4 Maya Guru1 Arjun Guru2 Ainaz Eftekhar2 Karen Farley1 Donovan Clay2 Jiafei Duan1,2 Piper Wolters1 Alvaro Herrasti1 Ying-Chun Lee2 Georgia Chalvatzaki5 Yuchen Cui3 Ali Farhadi1,2 Dieter Fox1,2 Ranjay Krishna♥1,2

### arXiv:2602.11337v2[cs.RO]19Feb2026

1Allen Institute for AI, 2University of Washington, 3University of California, Los Angeles, 4University of California, Berkeley, 5Technische Universität Darmstadt

*denotes equal contribution in no particular order. ♥ marks core contributors. See full author contributions here.

Data: Assets and Scenes Code: https://github.com/allenai/molmospaces Blog: https://allenai.org/blog/molmospaces

[Figure 1]

###### Abstract

[Figure 2]

Deploying robots at scale demands robustness to the long tail of everyday situations. The countless variations in scene layout, object geometry, and task specifications that characterize real environments are vast and underrepresented in existing robot benchmarks. Measuring this level of generalization requires infrastructure at a scale and diversity that physical evaluation alone cannot provide. We introduce MolmoSpaces, a fully open ecosystem to support large-scale benchmarking of robot policies. MolmoSpaces consists of over 230k diverse indoor environments, ranging from handcrafted household scenes to procedurally generated multiroom houses, populated with 130k richly annotated object assets, including 48k manipulable objects with 42M stable grasps. Crucially, these environments are simulator-agnostic, supporting popular options such as MuJoCo, Isaac, and ManiSkill. The ecosystem supports the full spectrum of embodied tasks: static and mobile manipulation, navigation, and multiroom long-horizon tasks requiring coordinated perception, planning, and interaction across entire indoor environments. We also design MolmoSpaces-Bench, a benchmark suite of 8 tasks in which robots interact with our diverse scenes and richly annotated objects. Our experiments show MolmoSpaces-Bench exhibits strong sim-to-real correlation (R = 0.96, ρ = 0.98), confirm newer and stronger zero-shot policies outperform earlier versions in our benchmarks, and identify key sensitivities to prompt phrasing, initial joint positions, and camera occlusion. Through MolmoSpaces and its open-source assets and tooling, we provide a foundation for scalable data generation, policy training, and benchmark creation for robot learning research.

###### 1 Introduction

Recent advances in robot learning [1–4], have given rise to increasingly general, open-vocabulary policies, capable of zero-shot deployment. As we work towards generalist robots, it becomes important to consider how to evaluate and measure the performance of these policies. State-of-the-art models are already nearing saturated performance on several established tasks, providing little signal to drive further progress [5]. Moreover, most manipulation benchmarks frequently focus on short-horizon skills in a single scene, failing to probe the long-horizon, compositional challenges

###### MolmoSpaces

High-Fidelity Physics Interaction

| | |
|---|---|
| | |

[Figure 3]

Real-World Success

| |
|---|

80%

[Figure 4]

60%

[Figure 5]

40%

[Figure 6]

20%

0%

0% 20% 40% 60% 80%

MolmoSpaces-Bench Success

[Figure 7]

[Figure 8]

[Figure 9]

###### Robots 130k Objects

42M Grasps

[Figure 10]

Asset Files Text Descriptions Scale & Mass

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

Comprehensive Benchmark

Physics Tuning Environment Testing

[Figure 18]

[Figure 19]

[Figure 20]

Experiment Tooling

Multiple Grippers & Embodiments

Object Model

Task Definitions

100g

- Figure 1 MolmoSpaces is an open ecosystem consisting of a large number of simulation environments, 3D articulated objects, and tasks for training and evaluating robot navigation and manipulation at scale. It provides object metadata, grasps, and tooling to generate training data, create benchmarks, and evaluate policies in a manner that correlates with real-world performance.

that arise in realistic environments [6–11].

The real world presents an extraordinarily long tail of situations a robot must handle. Kitchens vary in layout, lighting, and clutter. Objects come in countless shapes, sizes, and materials. Instructions can be phrased in myriad ways. A truly generalist policy must be robust to not just the common cases but to the vast combinatorial space of environments, objects, and tasks that constitute everyday life. Estimating a policy’s ability to do so requires evaluating on a far broader distribution of tasks, environments, and objects than ever before.

Simulation offers a compelling path to enable this level of rigor and scale in evaluation. Unlike physical experiments, which are expensive, slow, and difficult to reproduce, simulation enables systematic assessment across thousands of controlled scenarios. Rather than testing on a handful of cherry-picked scenarios, we can characterize policy performance across the full distribution of environments a robot might encounter. However, effective simulation for mobile manipulation must simultaneously support scene-scale diversity, physical realism, articulated interactions, and long-horizon compositional tasks in realistic indoor environments. For simulation experiments to be useful, results in simulation must attain strong correlation with real-world performance [12]. However, existing simulators and benchmarks remain limited. Many provide only dozens of scenes or objects, lack realistic physics or visuals, or support a narrow range of tasks.

We introduce MolmoSpaces, an end-to-end large-scale ecosystem for robotics research illustrated in Figure 1. MolmoSpaces unifies diverse scenes, objects, tasks, and tools for training and evaluating generalist robot policies. It contains over 230k diverse indoor environments spanning a wide range of layouts and scene types, which enables evaluation across the long tail of real-world spatial configurations. It also includes more than 130k high-quality rigid and articulated object models with rich semantic and physical metadata, which supports assessment of generalization to novel objects. In addition, MolmoSpaces provides over 42M annotated grasps across 48k interactive rigid and articulated objects, which offers ground-truth supervision for grasp success evaluation. Our assets and scenes dataset can be loaded into multiple simulators (MuJoCo [13], IsaacSim [14], and ManiSkill [15]), all backed by high-fidelity physics.

Using this ecosystem, we construct MolmoSpaces-Bench, a new benchmark suite that evaluates robot policies on 8 base tasks–navigate-to, pick, pick-and-place, pick-and-place-next-to, pick-and-place-color, open, close, and open-door (Sec. 4.1)– in never-before-seen environments, all zero-shot (i.e. with no fine-tuning on benchmark data). Importantly, the entire MolmoSpaces platform is open-source and extensible. This means that beyond the benchmarks we report, researchers can leverage MolmoSpaces to synthesize their own datasets of scenes, objects, and tasks for training robust robotic policies at scale. We hope that by providing a community-driven ecosystem of this scope, we will accelerate

###### Feature Molmo- Robo- AI2- Habitat iGibson RL- Behavior robo- Mani- OPTIMUS LIBERO Mimic-

Spaces Casa THOR 2.0 2.0 Bench 1K mimic Skill 2 Gen Scenes 232k 120 – 1 15 1 50 3 – 4 20 1 Objects 130k 2509 3578 169 1217 28 9318 15 2144 72 – 40 Object Categories 2.8k 153 – 46 – 28 1949 – – – – – Tasks 8 100 – 3 6 100 1000 8 20 10 130 12 Realistic Physics ✓ ✓ × × ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ Realistic Rendering (✓) ✓ ✓ ✓ × × ✓ × ✓ ✓ × × Muli-Embodiment ✓ ✓ ✓ × ✓ × ✓ × × ✓ × ✓ Room-Scale Scenes ✓ ✓ ✓ ✓ ✓ × ✓ × × × × × Multi-Room Scenes ✓ × ✓ ✓ ✓ × ✓ × × × × × Annotated Object Grasps ✓ × × × × × × × × × × × Mobile Manipulation ✓ ✓ ✓ ✓ ✓ × ✓ × ✓ × × ✓ Scripted Datagen ✓ ✓ × × × ✓ × ✓ ✓ ✓ × ✓ AI-generated Tasks ✓ ✓ × × × × × × × × × ×

- Table 1 Comparison to popular simulation frameworks used in the robot learning literature. The definition of tasks varies strongly across papers; many take it to be unique verb-object or category combinations.

progress toward truly general-purpose robotic intelligence.

In zero-shot evaluations (no task-specific fine-tuning), our benchmark distinguishes performance among several state-ofthe-art policies, including VLA models (π-models [1, 2, 16]) and classical modular baselines, across a wide range of unseen environments and objects. The results reveal steady progress over model generations, but also expose brittleness to distribution shifts. For instance, we find that minor changes in instruction phrasing or initial robot pose can cause significant drops in success for some policies, especially earlier-generation VLAs. This sensitivity highlights the importance of training on more diverse data, which MolmoSpaces can readily supply for future work. Encouragingly, we observe a strong sim-to-real correlation: policies that score higher in our simulation benchmarks also achieve better real-world success rates on equivalent tasks (with Pearson R2 ≈ 0.92 for object picking). This validates that high-fidelity simulation can be a reliable proxy for real-world performance. Moreover, by systematically perturbing scene parameters and sensor inputs in simulation, we pinpoint specific failure modes (e.g. dependence on certain camera viewpoints and lighting conditions) that are costly to uncover with physical trials. These analyses demonstrate how an open ecosystem like MolmoSpaces not only measures overall progress but also yields insights to drive algorithm improvement.

By dramatically expanding the scale of available simulated environments and making them openly accessible, MolmoSpaces enables researchers to measure generalization more rigorously than ever before and can support future work in generating diverse training data for tackling the next generation of robotics problems.

###### 2 Related work

Robot simulation frameworks provide scalable, safe, and repeatable platforms for rapid prototyping, policy learning, and evaluation. Modern simulators such as MuJoCo, Isaac, and ManiSkill offer high-fidelity physics simulations that support contact- and force-based manipulation [13–15, 17]. Building on these engines, the research community has developed a range of simulation frameworks. Projects like AI2-THOR [18], Habitat, and Habitat-Lab [19, 20] emphasize photorealistic visual navigation, but provide only limited manipulation support, often relying on “magic grasps” that bypass realistic contact dynamics [18, 21]. RoboCasa and RoboCasa365 [22, 23] build on MuJoCo to provide single-room environments, synthetic datasets, and task definitions for multi-task manipulation and navigation, but remain limited in scene and asset diversity.

Large-scale datasets in robotics are increasingly important, as demonstrated by efforts such as Open X-Embodiment [24].

In parallel, the community has pursued data scaling through simulation. Objaverse [25] provides internet-scale 3D assets compatible with simulators, while ProcTHOR expands AI2-THOR with tens of thousands of procedurally generated multi-room houses [21]. Holodeck [26] introduces LLM-guided scene generation beyond household environments, and InternScenes [27] combines real scans, procedural layouts, and designer-created environments to provide diverse indoor scenes at scale. Despite addressing scene and asset scale, these datasets offer limited support for physics-based manipulation and are primarily evaluated on navigation tasks. Conversely, GraspGen [28] provides large-scale grasp annotations for Objaverse assets, but these remain at the asset level and require substantial effort to integrate into interactive scenes.

By contrast, MolmoSpaces addresses both scale and task diversity through a ready-to-use ecosystem that unifies 230K scenes from AI2-THOR [18], ProcTHOR [21], and Holodeck [26], and makes them compatible across MuJoCo, IsaacSim, and ManiSkill. The ecosystem further incorporates 130K object assets from Objaverse [25], with over 48K objects annotated with grasp data and validated to be pickable and articulable under realistic physics. Together, these components enable scalable, diverse, and physically grounded evaluation of navigation, manipulation, and mobile manipulation policies. Comparisons between MolmoSpaces and prior work are summarized in Table 1.

Benchmarks are central to progress in robotics, providing standardized tasks and evaluation protocols for fair comparison, reproducibility, and diagnosis of failure modes. However, real-world benchmarking remains difficult to scale due to hardware heterogeneity, differences in sensing and control stacks, and the time and labor required for evaluation. Recent efforts such as RobotArena [29] partially address these challenges through distributed, crowd-sourced evaluation, while others like AutoEval [30] leverage success classifiers and reset policies to facilitate near-autonomous real-world evaluations of specific tasks. However, real-world evaluation alone remains limited in the scale and diversity needed to robustly assess generalist policies.

Simulation-based benchmarks offer a scalable and reproducible alternative, enabling controlled variation and systematic stress testing that are impractical in the real world. A wide range of benchmarks have emerged for manipulation [6, 8] and navigation [18, 31, 32], becoming standard testbeds for evaluation. RoboVerse [33] unifies several of these benchmarks under a shared framework. As policies converge on vision–language–action (VLA) models, recent benchmarks such as LIBERO [6], CALVIN [7], LIBERO-Plus [9], LIBERO-Pro [10], VLABench [34], and RobotArena-Infinity [11] expand evaluation to include language grounding and generalization. Additional works like the COLOSSEUM [35] and VLATest [36] evaluate the robustness of generalist policies to changes in lighting, distractor objects, camera poses, and more. Other efforts, including BEHAVIOR-1K [37], ManiSkill-HAB [38], and EmbodiedBench [39], extend benchmarks to mobile manipulation, but remain biased toward household environments and limited in scene scale.

Sim-to-real benchmarks focus on providing simulation evaluations that match real-world results. SIMPLER [40] studies distributional shifts by pairing simulated evaluations with real-robot rollouts, while PolaRiS [12] constructs digital twins from real-world videos and demonstrates strong sim-to-real correlation across realistic scenes, albeit limited to tabletop manipulation and partial environment reconstruction. Other works analyze large behavior models across simulated and real environments, but rely on proprietary evaluation pipelines and closed datasets. Moreover, many sim-to-real benchmarks require training on simulation data, limiting their ability to assess true zero-shot generalization [12, 37].

In contrast, our benchmark evaluates zero-shot generalist policies across navigation, manipulation, and mobile manipulation tasks using a validation set of over 20K diverse indoor scenes—spanning both household and non-household environments—and more than 22K interactable rigid and articulated objects. It supports large-scale distributional analysis under controlled perturbations to scenes, objects, sensors, and language prompts, enabling rigorous assessment of generalization and failure modes.

###### 3 MolmoSpaces

MolmoSpaces contains:

- 1. MolmoSpaces-Scenes: Over 230k diverse indoor environments spanning a wide range of layouts and scene types, enabling evaluation across the long tail of real-world spatial configurations.
- 2. MolmoSpaces-Objects: More than 130k high-quality rigid and articulated object models with rich semantic and physical metadata, supporting assessment of generalization to novel objects.
- 3. MolmoSpaces-Grasp: Over 42M annotated grasps across 48k interactive rigid and articulated objects, providing ground-truth supervision for grasp success evaluation.
- 4. MolmoSpaces-Bench: A benchmark suite comprising object-centric tasks across 8 task types, zero-shot evaluations with no fine-tuning, and sim-to-real correlation analysis demonstrating that lessons learned in simulation transfer to real-world performance.
- 5. Simulation Infrastructure: Scalable tooling for task composition, benchmark creation, and reproducible evaluation.

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

- Figure 2 Examples of diverse scene environments from MolmoSpaces-Scenes-MultiType with the Filament rendering engine. Our ecosystem contains scenes from art studies, cat cafes, lounges, museums, and many other scenes, all pre-populated with objects to be manipulated.

###### Dataset Name Stab. Lift Inter. Artic.

Dataset Scene Object Obj / Creation Name Count Set Scene Method

MSCrafted 100.0 97.5 98.3 99.1 MSProc 99.4 99.7 99.9 97.4 MSProcObja 98.4 98.7 99.1 93.7 MSMultiType 94.9 99.6 93.7 65.2

MSTwin 1 - 8 Manual MSCrafted 120 THOR ∼30 Hand-crafted MSProc 12k THOR ∼72 Heuristic MSProcObja 110k THOR+ ∼60 Heuristic MSMultiType 110k THOR+ ∼97 LLM Proc.

- Table 3 Scene datasets are all quality tested in the MuJoCo simulator with high pass rates.

Dataset Name MSCraft MSProc MSObja MSMulti Stability test (%) 92.5 93.9 95.9 95.2

- Table 4 Isaac Sim simulator pass rates.

- Table 2 MolmoSpaces-Scenes contains five scene datasets with varying scene types, scales, objects, and creation methods.

###### 3.1 MolmoSpaces-Scenes

We provide five scene datasets, summarized in Table 2. The scenes were originally sourced from multiple datasets in AI2-THOR: MolmoSpaces-Scenes-Crafted, MolmoSpaces-Scenes-MultiType, MolmoSpaces-Scenes-Procedural, and MolmoSpaces-Scenes-Procedural-Obja. We process and tune these scenes to be physically realistic in MuJoCo, ManiSkill, and IsaacSim. MolmoSpaces-Scenes-Crafted (MSCrafted) contains 120 hand-crafted single-room scenes evenly distributed among kitchens, bedrooms, living rooms, and bathrooms, split into 48/48/24 train/validation/test scenes, with object placement carefully curated to ensure physical stability. MolmoSpaces-Scenes-Procedural (MSProc) provides 12k procedurally generated residential scenes with 10k/1k/1k train/validation/test splits, where each scene represents a house with between one and ten rooms with layouts designed for realistic household navigation and manipulation. MolmoSpaces-Scenes-Procedural-Obja (MSProcObja) extends this with 110k train/validation scenes containing both THOR and Objaverse objects. MolmoSpaces-Scenes-MultiType (MSMultiType) provides 110k diverse scene types generated via LLM-based procedural generation, also with THOR and Objaverse objects. Finally, MolmoSpaces-Scenes-DigitalTwin (MSTwin) is a high-fidelity manual reconstruction of our real-world kitchen. All datasets contain both rigid and articulated objects. Some examples of these scenes are shown in Figure 2.

To generate MSMultiType, we extend the diverse scene generation pipline presented in Holodeck [26]. We select and extend indoor scene types in the SUN database [41], based on the suitability to the available THOR and Objaverse object taxonomy, and reorganize them into a hierarchy with between five and fifteen concrete scene types for each of ten generic types. In total, 546 room types (including many scene-specific) and 101 scene types are available for scene sampling (Fig. 17 left, in the appendix). Each scene specification contains a generic or concrete scene type and between one and ten rooms of diverse types. We also sample from a subset of 52k persona descriptions from [42] – chosen by their suitability to produce visual and stylistic differences in objects, materials, or layout constraint selection – and accentuate some particular style in 90% of the scene specifications, which are finally converted to text prompts for LLM scene generation. We add a ‘grid’ constraint to the DFS-based object placement optimizer in Holodeck to simplify the

[Figure 29]

###### Figure 3 An example scene rendered across different simulators: MuJoCo, Issac Sim, and ManiSkill. When using MuJoCo, the scenes can be rendered using either the OpenGL renderer (Classic) or with Filament (Filament).

uniform placement of objects in available free space commonly occurring in non-residential scenes.

In order to enable the wide-spread use of our simulation assets, we also release them converted to the USD format, which can be natively loaded in IsaacSim. Additionally, we provide both an asset and scene loader for ManiSkill. Figure 3 illustrates the same scene rendered across aforementioned simulators and their respective rendering engines. Finally, we generate occupancy maps for all scenes to identify collision-free starting poses for robots, ensuring safe initialization for both manipulation and navigation experiments.

Scene quality testing: For an environment to be suitable for both navigation and manipulation tasks, objects must remain physically stable, respond appropriately to applied forces, and be accessible for interaction. Specifically, objects should not drift or randomly move, rigid objects must be pick-upable, articulated objects must allow motion across their defined ranges, and masses, friction, and other physical parameters should yield realistic dynamics under standard forces. Ensuring these properties is particularly important when integrating assets not originally designed for physics-based simulation. Additionally, objects that are originally designed for such simulations often have innately incompatible parameters by default which also warrant additional tuning for integration.

To systematically enforce these conditions, we implemented four tests addressing different aspects of physical validity: scene stability, object intersections, rigid object liftability, and articulation of movable objects. First, each scene was settled by simulating approximately 20 seconds and then saved with the updated, settled poses. Stability tests simulate the environment for multiple steps after settling and remove any objects that continue to jitter. Intersection tests detect colliding objects: if a collision occurs between a fixed and a free object, the free object is removed; if the collision is between two free objects, the smaller one is removed. Lift tests apply upward forces to all free objects and measure their displacement along the z-axis; objects that cannot be lifted by at least 5 cm and are detected inside another object’s site are removed. Articulation tests apply forces at joints to open or close articulated objects; if an object cannot be actuated through at least 70% of its joint range, free objects located within its articulation site or blocking its motion are removed

Across MuJoCo scenes (Table 3), over 95% of environments pass these stability and manipulation checks. The lowest success rate is observed for articulation tests on the MSMultiType scene dataset (63%). We note that these scenes were designed to maximize navigability by biasing large object placement to prevent blocking door-to-door navigation while ensuring many small objects can be accessibly placed for manipulation tasks. Manual inspection confirms that most failures arise from scene layout rather than limitations of the simulation engine. We additionally evaluated a subset of scenes in Isaac Sim, observing consistently high pass rates as shown in Table 4, which demonstrates the robustness and cross-platform validity of our validation pipeline.

- 3.2 MolmoSpaces-Objects

We provide two object model datasets consisting of 1.6k THOR and 129k Objaverse objects, with samples shown in

###### Figure 4. These assets populate the generated scenes described in Section 3.1. Table 5 reports the average number of objects per scene for each dataset. To ensure physical realism, rigid objects were validated by estimating mass and density against LLM-estimated values, and articulable objects were tuned by manipulating them via teleoperation of a simulated Franka FR3 robot. Collider meshes were generated using COACD [43], with primitive colliders human annotated for all THOR assets. For stability, receptacle objects primarily use primitive colliders, while manipulable objects use convex decomposition except for very small or thin items, where primitives are preferred. Meshes in

Dataset Total Pickupable Non-Pick Articulated THOR Objaverse

MSCrafted ∼30 ∼14 ∼7 ∼9 ∼23 – MSProc ∼72 ∼34 ∼31 ∼7 ∼41 – MSProcObja ∼105 ∼47 ∼50 ∼7 ∼72 ∼32 MSMultiType ∼150 ∼61 ∼77 ∼12 ∼73 ∼77

- Table 5 Average number of objects per scene on each dataset

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

- Figure 4 A random sampling of object types in our ecosystem, with different sizes, shapes, and articulations. These examples are rendered with Filament.

Objaverse models were further processed and decimated for simulation efficiency [44].

THOR objects span 134 categories, with 22 articulable categories (e.g., doors, refrigerators) annotated with joint types, axes, positions, and ranges. Objaverse objects, spanning almost 2.8k WordNet [45, 46] synsets, are curated from 625k models annotated with descriptions, mass estimates, canonical poses, pickable and receptacle properties, synsets, and scale estimates generated by GPT-4o [47, 48]. A complementary GPT-4.1 [49] annotation identified object counts, extraneous or missing geometry, texture quality, and receptacle presence in models with parseable annotation. Filtering provided 129k single-object models that met scale consistency, sufficient texture quality, cross-renderer fidelity, compact file size, collider quality, and synset coverage as objects to be placed in MSMultiType scenes. Additional filtering was done for MSProcObja to keep only objects with synsets heuristically mapped to placement-compatible THOR categories, resulting in 92k objects across 2k synsets. Further details in filtering and curating objects from Objaverse are described in Appendix A.

Object models are accompanied by extensive physical and semantic metadata, convex colliders, and canonical coordinate definitions, besides grasps, which are obtained as described below. To support easy integration into robotics simulation workflows, all object models are provided in formats compatible with MuJoCo, IsaacSim, and ManiSkill.

###### 3.3 MolmoSpaces-Grasp

We introduce a comprehensive grasp dataset, MolmoSpaces-Grasp, consisting of over 42 million grasps that cover objects in MolmoSpaces scenes. Our dataset provides 6-DoF grasp poses for two types of manipulable objects: rigid objects, which can be picked and moved as single bodies, and articulable objects, whose constituent parts can move relative to one another via a revolute or prismatic joint. Our grasp generation process builds on prior work such as

| |[Figure 49]|
|---|---|
| | |

[Figure 50]

###### Final Grasp Dataset

Antipodal Sampling

Rigid 3D Object Models

[Figure 51]

(40M+ Grasps)

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

Shaking Test

[Figure 60]

[Figure 61]

Articulated 3D Object Models

[Figure 62]

[Figure 63]

[Figure 64]

Leaf Mesh Extractor Articulation Test

- Figure 5 Our grasp generation pipeline consists of separate streams for rigid and articulated assets. We generate 42M+ verified grasps that can be utilized to create scripted interaction policies. Grasps can be used in different simulation environments, with an Issac example shown on the right.

GraspGen [28], 6-DOF GraspNet [50] and ACRONYM [51], with extensions to support articulable objects through a new functionality-based evaluation step. We apply this pipeline to 48,111 objects drawn from a curated subset of Objaverse and custom-designed THOR assets, using separate pipelines for rigid and articulated objects to reflect their distinct manipulation requirements. Table 8 compares our dataset with recent large-scale grasp datasets.

Grasp generation: Our pipeline (Figure 5) begins with 3D object models, from which we extract mesh colliders for grasp sampling. Antipodal contact pairs are sampled based on the geometry of the Robotiq 2F-85 gripper and the object. For rigid objects, sampling is performed across the full mesh surface, whereas for articulated objects, it is restricted to leaf components corresponding to handles or other functional interaction points. Grasps that result in collisions with non-leaf geometry are immediately discarded.

From the initial samples, we select up to 1,000 diverse and robust grasps per object. To ensure diversity, sampled grasps are clustered in the full 6-DoF pose space and tested uniformly across clusters. Grasp robustness is evaluated differently depending on the object type. For rigid objects, we apply controlled linear and rotational perturbations to the gripper upon grasping and discard any grasp that fails to maintain contact, leading to object slippage. For articulated objects, we define a grasp robustness by its actuation feasibility, meaning it can actuate the relevant joint through at least 70% of its valid range in both directions, while maintaining stable contact.

To evaluate the practical utility of annotated grasps in MolmoSpaces scenes, we perform in-situ tests using a floating Robotiq 2F-85 gripper. Candidate grasps that collide with scene geometry are discarded, with collision checks performed at the pre-grasp pose (4 cm offset along the gripper’s negative local z-axis), grasp pose, and along the execution trajectory. The gripper then executes the pre-grasp and grasp motions, followed by lifting the object or articulating its moving part along the object’s joint path. Success rates (Fig. 24 in Appendix) are reported only for grasps that pass all collision checks and complete the intended motion.

Initial evaluations showed that objects placed on surfaces often had few viable, non-colliding grasps. This is because grasp generation is performed on isolated objects without surrounding geometry and ignores gravity, resulting in high failure rates due to slippage during in-situ testing. To mitigate this, we updated the pipeline to generate more robust and stable grasps by biasing contacts toward the center of the fingertips as illustrated in Figure 6 . For small or thin objects, such as forks and pens, fingertip-edge contacts are preferred.

Remaining failures are largely due to scene- and object-level constraints: for lifting, common issues include objects too large for the lift pose, confined spaces limiting vertical clearance, collisions along the execution trajectory, and object slippage through the grasp; for articulated objects, failures occur when the object lifts instead of actuating, collisions exists along the articulation path, obstacles block articulation, or the gripper misaligns with the handle. These results underscore the importance of in-situ evaluation for producing grasps that are functional and practically useful given the way objects contextually exist within the scene.

[Figure 65]

Figure 6 RobotIQ 2F-85 gripper and preferred grasping locations for sampling grasps

# Experiment

𝜋

##### Scenes Objects Robots Tasks Cameras

Policies

- Figure 7 Code Structure with modular experiment composition.

###### 3.4 Robots

We provide robots with varying amounts of mobility and complexity, covering the spectrum of widely-used manipulation platforms. We categorize these platforms as static or mobile, and single-arm or bimanual. To handle all of these cases, we provide a Franka FR3 arm with three gripper configurations (Franka Hand, RobotIQ 2F-85, and CAP) as a static manipulator, Rainbow RB-Y1 as a bimanual and holonomic mobile manipulator, and floating CAP [52] and RobotIQ grippers, unconnected to arms, so without kinematic limits. The Franka FR3 with the RobotIQ 2F-85 gripper is specifically set up as a DROID [53] system, with corresponding cameras with the correct intrinsics and extrinsics. The Franka FR3, RobotIQ Gripper, and Rainbow RB-Y1 robot models were sourced from [54].

Robot Control: For manipulators, our framework provides for both absolute and relative joint position commands, which are tracked internally with a gravity-compensated joint-space stiffness controller. Mobile platforms, such as the holonomic RB-Y1 base or 6DoF floating CAP, can be controlled via absolute or relative poses.

Kinematics Computation: We provide built-in forward and inverse kinematics solvers for each robot, and the modular nature of our framework makes it easy to further extend for new robots. Our parallelized inverse-kinematics solver is written in JAX, and leverages Levenberg-Marquadt optimization with null-space control for posture regularization. This solver can natively be GPU-accelerated, but even on a CPU can solve batches as large as 256 samples with high precision in ∼200ms. For the RB-Y1 robot, which is configured for use with the cuRobo[55] motion generator, we provide forward and inverse kinematics as a wrapper around cuRobo’s functionality.

###### 3.5 Modular experiment composition

MolmoSpaces supports modular experiment composition by flexibly combining scenes, tasks, robots, and camera configurations, as illustrated in the Figure 7. We provide camera setups for commonly used RealSense, ZED, and GoPro cameras. Beyond vision inputs—including calibrated multi-view RGB and optional depth cameras with object image points—the framework exposes rich proprioceptive signals (e.g., joint states, end-effector and base poses), task-state information (object and articulation states), as well as task annotations, planner signals, and action histories. Together, these sensors capture the full interaction context.

###### 3.6 Data collection

Two different modes of data collection are possible. Manual data collection is possible using TeleDex [56] iPhone app, which uses iOS ARKit to stream the phone’s pose. In addition to this, leveraging the pre-computed grasps, it would be possible for scripted policies to control the robot to solve the defined tasks.

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

a) “open the oven" b) “close the drawer" c) “pick up the cup" d) “place the spray in the bowl"

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

e) “pull" (door) f) “find a lamp" g) “place pencil next to vase" h) “prepare a simple meal"

- Figure 8 Example images of our range of tasks, spanning from manipulation of articulated and non-articulated assets to navigation and long-horizon tasks, shown together with their associated text instructions. These examples are from the MuJoCo simulator.

###### 4 Benchmark

To enable rigorous and reproducible evaluation of robot policies, we introduce MolmoSpaces-Bench, which spans eight base tasks across navigation, manipulation, and mobile manipulation. These benchmarks are designed with explicit diversity requirements across scenes, object categories, and robot configurations, with each trial verified for solvability.

###### 4.1 Tasks

Leveraging the diverse scenes, assets, and robots provided by MolmoSpaces, we introduce a suite of tasks and corresponding benchmarks designed for comprehensive policy evaluation. We define eight base tasks: navigate-to, pick, pick-and-place, pick-and-place-next-to, pick-and-place-color, open, close, and open-door. For each of these tasks, we also provide a well-defined success condition and a dense reward function.

- 1. Navigate-to: A policy must search for and navigate to a specified target object. The robot is initialized between 4 and 20 meters away from the target object, possibly in an entirely different room. The success conditions require the object to be visible from the robot’s head camera and closer than 1.5 meters away. We sample the same object candidate set as [57].
- 2. Pick: Grasp and lift a specified object from its initial location by at least 1cm.
- 3. Pick-and-place: Move a target object to be into or onto a target receptacle. To be counted as successful, at least 50% of the object’s weight must be vertically supported by the receptacle. Additionally, the target receptacle cannot have been displaced by more than 10 cm or 45 ° from its initial pose.
- 4. Pick-and-place-color: Similar to pick-and-place, with the same initialization and success conditions. However, the task is complicated by multiple duplicates of the target receptacle differentiated only by color. This color is included in the task prompt, requiring policies to attend to and follow specific task instructions.
- 5. Pick-and-place-next-to: Move a target object to be next to a target receptacle. The target receptacle is initialized 30-50 cm away from the target object, and to be successful, the policy must move the object to be closer than 5 cm surface-to-surface from the target receptacle. The object must additionally be on the same support surface (e.g., table) as the receptacle, which cannot have been displaced by more than 5 cm or 45° from its initial pose.
- 6. Open: Open an articulated household fixture such as a drawer, cabinet, microwave, or fridge. Starting fully closed, the robot must open the fixture by at least 15%.
- 7. Close: Close an articulated household fixture, which is initialized halfway open. To succeed, the policy must close the fixture to at most 15% open or 85% closed.
- 8. Open-door: Open a hinged door by manipulating its handle or surface. The door starts fully closed, and the policy is tasked with opening it by at least 67%.

Navigation The navigate-to task is evaluated with the Rainbow RB-Y1, which is instructed to locate and navigate to a given object. Following [57], the robot is initialized 4-20m away from the target object, and success is defined as being closer than 1.5m to the target object with it clearly visible in the navigation camera. For this task, policies must explicitly signal task completion, and incorrectly doing so is counted as a failure.

Rigid-body manipulation Non-articulated manipulation includes the pick, pick-and-place, pick-and-place-color, and pick-and-place-next-to tasks, which are evaluated with a Franka FR3 robot in the DROID configuration. For pick, the robot must grasp and lift an object by at least 1cm. For pick-and-place, the robot must move an object into or on a target receptacle, while pick-and-place-next-to requires the robot to place the object next to the receptacle. The pick-and-place-color task is a variant of pick-and-place with multiple similar but differently-colored receptacles, requiring policies to attend and adhere to specific task instructions.

Articulated manipulation Our three articulated manipulation tasks cover both static and mobile manipulation.The open and close tasks are static manipulation tasks evaluated with the FR3 in the DROID setup, where the robot must open or close a variety of articulated household fixtures, including cabinets, drawers, refrigerators, and microwaves, by at least 15%.The open-door task is a mobile articulated manipulation task, where an RB-Y1 robot must push or pull a door to be at least 67% open, requiring coordinated mobility and manipulation across many degrees of freedom.

Specifically, we evaluate following models:

- 1. PI models (manipulation). We evaluate generalist models from the PI family, namely π0, π0-FAST, and π0.5. As these models are trained primarily on real-world data, this setting constitutes a real-to-simulation evaluation. We follow the DROID hardware setup for this set of evaluations.
- 2. CAP models (manipulation). We evaluate CAP models that are task-specific and available for the following tasks: pick, open, and close. The models are conditioned on 3D “contact” points, which are provided by using Gemini-Robotics-ER-1.5 [58] at the initial step of the episode. We use the custom gripper design that CAP policies are trained with to replace the Robotiq Gripper from the DROID setup.
- 3. RING model (navigation). RING is an embodiment-agnostic indoor navigation policy trained entirely in simulation using large-scale randomization over robot body geometry and sensor configurations.
- 4. DualVLN (navigation). DualVLN is a dual-system VLN foundation model that separates high-level reasoning from low-level control: a VLM-based global planner predicts mid-term waypoint goals, with a lightweight diffusion-based policy that executes smooth, real-time trajectories conditioned on these goals.

###### 4.2 Benchmark creation

For every task, we provide one or more benchmarks designed for comprehensive policy evaluation. These benchmarks draw from multiple scene datasets, described in Sec. 3.1, and provide varying levels of difficulty and complexity. Concretely, each benchmark is defined by an initial scene, robot, and camera configuration, as well as a descriptive task instruction. To ensure benchmark quality, we perform balancing to maximize diversity of object categories and instances, as well as scenes. Additionally, each of our provided benchmarks are guaranteed to be solvable with the provided robot and initial conditions, ensuring task feasibility.

We generate a full set of benchmarks with all combinations of environments and tasks, listed in Table 6. For easier comparisons, we also include preferred evaluation configurations. For the open and close tasks, we choose the MSCrafted scenes, as these contain handcrafted kitchens with many cabinets and drawers. For most manipulation tasks, we choose the MSProcObj variants, as the presence of objaverse assets gives object diversity. For the door-opening task, we use the MSProc environment, as object diversity is irrelevant. All tasks have 2k samples. We present select benchmark results in Sec. 5.1.

###### 4.3 Extensions

Controlled variants of benchmarks enable easier testing of hypotheses. For example, to answer the question “Does a policy perform as well on open vocabulary navigation as on closed vocabulary navigation?", one can evaluate on MSProc, which uses a closed set of object categories, versus MSMultiType, which supports open vocabulary. In addition to this, MolmoSpaces also allows for the easy creation of controlled adversarial benchmarks to test specific functionality,

###### Task MSCrafted MSProc MSProcObj MSMultiType

Open (Franka) ✓+e ✓ ✓ ✓ Close (Franka) ✓+e ✓ ✓ ✓ Pick (Franka) ✓ ✓+e ✓ ✓ Pick & Place (Franka) ✓ ✓+e ✓ ✓ Pick & Place Next To (Franka) ✓ ✓ ✓ ✓ Door Open (RBY1) ✓ ✓ ✓ ✓ Pick (RBY1) ✓ ✓ ✓ ✓ Pick & Place (RBY1) ✓ ✓ ✓ ✓ Navigation (RBY1) ✓ ✓

- Table 6 Availability of manipulation and navigation benchmarks across environments. ✓ indicates the benchmark is available, +e indicates there is an easy variant of the benchmark, ✓indicates the preferred evaluation environemnt.

such as the ability of a policy to handle varying initial robot configurations (shown in Fig. 12) or the ability to do manipulation in progressively more cluttered environments.

Additionally, while our provided benchmarks cover a variety of short-horizon tasks, long-horizon evaluation is also a critically important aspect of benchmarking. To that end, we provide an LLM-based task generation system that can generate long-horizon tasks with more abstract task descriptions. In this process, shown in Fig. 9, we take an existing scene and prompt an LLM to generate feasible long-horizon tasks, defined by a high-level task description and a sequence of base tasks as defined in Sec. 4.1. Unlike the base tasks, the tasks obtained by this sampling procedure can result in longer horizons, e.g., needing to open a fridge before taking objects out and placing them elsewhere.

[Figure 74]

|Parse Output & Convert to MultiTask Reward reward = sum(task_definitions[output[i]])| |
|---|---|
| | |

position, bounding box, relations, etc.

JSON

Prompt "Make a salad"

#### LLM

Task

Pick_Place(Tomato, Bowl), Open(Refrigerator), Pick_Place(Lettuce, Bowl),

pick, place, open, close

Tasks

- Figure 9 The LLM-based long-horizon task generation system makes use of text-form scene descriptions to generate new task descriptions and success condition checks based on predefined atomic checks.

###### 5 Experiments

###### 5.1 Evaluations

We use MolmoSpaces-Bench to evaluate open-source manipulation and navigation policies across the diverse scenes of MolmoSpaces zero-shot, without task-specific fine-tuning as done in many prior evaluations. We believe this setting is more practical, as it evaluates models as they are released and intended to be used, rather than their ability to efficiently fit additional task-specific data. We report the results under the standard settings in in Fig. 10.

###### 5.1.1 Manipulation tasks

We evaluate vision–language–action (VLA) models from the π family [1, 2, 16] (π0, π0-FAST, and π0.5, specifically the joint position variants that have been fine-tuned on the DROID dataset [12]) as well as utility contact-action models from CAP [52]. These models make use of different setups as described in Sec. 3.4, and therefore operate under different sensing and embodiment constraints. In particular, the DROID setup provides the π models with two camera views,

| |3%<br><br>14%<br><br>20%<br><br>28%<br><br>9%<br><br>16%<br><br>34%<br><br>55%<br><br>|
|---|---|
| | |
| | |
| | |
| | |
| | |
| | |

100

84%

SuccessRate(%),mean

80

62%

60

48%

47%

42%

35%

40

21%

18%

20

9% 10%

7% 10% 10%

0

Pick

Open

Close

Pick & Place

Navigation

CAP Policies π Policies Other CAP-EC1 CAP-EC2 CAP-EC3 CAP π-0 π-0 FAST π-0.5 Paligemma Binning DualVLN RING

- Figure 10 Zero-shot success rates of different baseline policies across five MolmoSpaces benchmark tasks. Showing expected performance improvement in improved policies. Error bars show 95% Bayesian credible intervals.

0% 20% 40% 60% 80% Benchmark Performance

0

20

40

60

80

100

Real-worldPerformance*(%)

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

R=0.96 [0.92, 0.98] ρ=1.00 [0.93, 1.00]

Pick

0% 20% 40% 60% 80% Benchmark Performance

0

20

40

60

80

100

Real-worldPerformance*(%)

| | |
|---|---|
| | |

| | |
|---|---|
| | |

R=0.85 [0.57, 0.99] ρ=0.40 [0.20, 1.00]

Open

0% 20% 40% 60% 80% Benchmark Performance

0

20

40

60

80

100

Real-worldPerformance*(%)

| | |
|---|---|
|| | |
|---|---|
| | |
| |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

R=0.97 [0.62, 1.00] ρ=1.00 [0.20, 1.00]

Close

CAP Policies π Policies Other

CAP CAP-EC1 CAP-EC2 CAP-EC3 π-0 π-0 Fast π-0.5 Paligemma Binning

- Figure 11 Sim-to-real correlation results for pick, open, and close task. Coefficient of determination (R) and the Spearman rank correlation coefficient (ρ) are shown.

whereas the floating CAP setup uses a single wrist-mounted camera and has no kinematic constraints. During benchmark construction, we ensure that all manipulation tasks are feasible under the DROID setup.

###### 5.1.2 Navigation Tasks

We select two state-of-the-art prior methods for visual object-goal navigation. RING [59] is an embodiment-agnostic, transformer-based navigation policy trained entirely in simulation that demonstrates robust generalization across a wide range of real-world robot platforms. It is trained on semantic navigation tasks where instructions consist of a simple verb (e.g., “go to", “locate", “find", “search for", “navigate to") paired with an object category. DualVLN [60] is a dual-system vision–language navigation (VLN) foundation model that integrates high-level reasoning with low-level action execution. Unlike RING, DualVLN is trained on detailed instructions that specify intermediate steps the robot must follow, rather than simple semantic goals (see Fig. 4 of [60] for examples). Both policies are evaluated on the navigate-to benchmark, which comprises 2,000 trajectories across 679 houses and goal objects drawn from 568 synset categories. This difference in task formulation explains the performance gap between policies (see Fig. 10), as our benchmark’s semantic navigation format poses a distributional mismatch for DualVLN while aligning with RING’s training. We adopt semantic navigation because it is more directly compatible with downstream mobile manipulation tasks.

###### 5.2 Sim-to-Real Correlation

Correlation between performance in real and simulated evaluations shows how predictive simulation results are of real world behavior. We therefore compare the results achieved by policies in our benchmarks to their real-world performance, which we take from RoboArena [29] and CAP [52]. We evaluate the correlation for the pick, open, and close tasks individually. Results for are shown in Fig. 11. For the pick task, we observe a strong linear correlation

50

###### SuccessRate(%)

40

29% 28% 27%

29% 28%

25%

24%

30

20%

40

20

SuccessRate(%),mean

29% 31%

10

28% 29%

27% 28% 27% 28%

30

−0.5 −0.3 0.0 0.3 0.5

0.5 1.0 2.0

20%

Joints Perturbation (rad)

17%

Light Multiplier (x)

20

14%

13%

61%

55%

Third-Person Wrist

SuccessRate(%)

60

10

40

0

28%

22%

23%

π-0 π-0 Fast π-0.5

20%

###### Droid Verb (Sequence Frequency)

20

19% 2%

pick→eat (0) pick (177) pick→place (1,657) put (25,985)

0.0 3.0 9.0

0.0 0.5 1.0

Figure 13 Prompt sensitivity: π models performance on the Pick-MSProc-1k benchmark with varying prompts. Frequent DROID prompts perform better.

Point Conditioning Noise (cm)

Camera Occlusion Probability

CAP π-0.5

- Figure 12 Robustness analysis under environmental perturbations. Top: Joint noise and lighting. Bottom: Point noise and camera occlusion.

between our MolmoSpaces-Bench results and the results from 752 RoboArena pick tasks, with Pearson and Spearman rank correlation coefficients of 0.96 and 0.98, respectively. This underlines MolmoSpaces-Bench’s utility and predictive power. As in the Pick task, we compute correlations between benchmark success rates and real-world performance measured on RoboArena and CAP for the open and close tasks. While fewer real-world episodes are available for these tasks, we observe consistent positive correlations but with bigger error bars.

###### 5.3 Distributional Evaluations

Machine learning models typically perform best under small distribution shifts between training and evaluation. Given that the π series models are trained on datasets that include DROID [53], we evaluate how performance degrades as we move away from this distribution. This evaluation is enabled by the scale of MolmoSpaces, which allows us to systematically vary environmental and policy factors beyond aggregate success metrics.

To probe language-induced distribution shift, we vary the pick task prompt using verb sequences of increasing frequency in DROID instructions, as shown in Figure 13. When using phrasing that are more frequent in the DROID dataset, π0 achieves a success rate within 1% of π0.5, compared to a 14% gap otherwise. With the assumption that the underlying DROID data distributions used to fine-tune π models are similar, this sensitivity to prompt phrasing suggests that generalization limitations of earlier π models arise from the language-conditioning component rather than the action heads. We observe this effect for the pick task in our benchmark and leave validation of its generality in other tasks to future work.

A similar degradation is observed when perturbing the initial joint positions of the π0.5 policy from the default configuration used in DROID. As shown in Figure 12, varying the starting joint positions away from this default leads to a decline in performance. In contrast, varying the lighting intensity has little effect on performance, likely because such visual distribution shifts are mitigated through image-based data augmentation.

Figure 12 also illustrates the effect of other environmental perturbations on the performance of policies. In particular, occluding the wrist camera reduces π0.5 success rate to 2%, while occluding the third-person camera only lowers the performance to 20%, indicating a strong reliance on wrist-mounted visual input. Similarly, CAP’s performance indicates a strong reliance on a good starting conditioning point. Figure 15 shows that π0.5 and CAP prefer different grasping approaches, with π0.5 favoring top-down grasps and CAP favoring side grasps. This preference helps explain why π0.5 performs better on objects with top openings, such as mugs and bowls, while CAP performs better than π0.5 on objects where side grasps are feasible, such as bottles and apples, as shown in Figure 14.

*The real-world performance is obtained from RoboArena by filtering for pick tasks and using RoboArena’s partial success criteria.

[Figure 75]

- Figure 14 Per-object category success rates for CAP and π policy families on the pick task. Colors are normalized per row within each table.

###### 5.4 Controlled Policy Comparison

To ensure a fair comparison between policies, we use a Franka FR3 arm setup as a unified embodiment for all manipulation benchmarks, alongside three RGB-D cameras: one wrist-mounted and two third-person. We allow a fixed offset in the robot base frame relative to the initial task configuration to accommodate grippers with differing geometries. This offset must be constant across episodes and should not use any privileged or task-specific information. During task generation, we filter out benchmark tasks that are not physically achievable under the standard DROID setup. We acknowledge that this filtering procedure may advantage grippers such as the RobotIQ 2F-85. We also acknowledge that the choice of end-effector can affect task difficulty, but consider this a part of system design to be evaluated.

Another variable that affects our recorded policies performance is the task horizon, which we set as 300 for the π models, and 50 for CAP. Figure 16 compares oracle termination with fixed-horizon termination of the π models, showing that policies sometimes reach a successful state but subsequently undo it before the episode ends. This behavior is also reflected in Table 7, where on average π0.5 goes through 2.65 grasp-ungrasp transitions before exceeding the reward threshold in successful episodes, while π0 makes 4.63 such transitions. These results suggest that π models benefit from sufficient time to retry actions, whereas CAP relies on VLM-based supervision for retries that is not used in our baseline benchmark experiments.

- Table 7 Average number of grasp-ungrasp transitions before task success for π policies on the pick task.

Policy Mean ± Std Median

π-0 4.63 ± 5.26 3.0 π-0 Fast 2.02 ± 2.62 1.0 π-0.5 2.65 ± 2.74 1.0

| |0°<br><br>| |
|---|
|CAP<br><br>Model|90°|
|---|---|---|---|
| | | | |

50

90°(SideGrasp)

0°(Top-Down)

π-0.5

40

Frequency

35%

SuccessRate(%),mean

28%

30%

30

25%

20%

18%

20%

20

14%

12%

15%

10%

10

10%

5%

0

0%

0 45 90

π-0 π-0 Fast π-0.5

###### Evaluation Type of Pick up Task

Grasping Approach

Fixed Episode Length (Can Undo Success) Oracle Termination (Stops at Success)

- Figure 15 Grasp direction histogram of successful grasp of the π0.5 and CAP policies.

Figure 16 Comparison of oracle termination and fixed-horizon termination.

###### 6 Conclusion

We present MolmoSpaces, a comprehensive open ecosystem comprising large-scale simulation environments, diverse object assets, and extensive grasp datasets. We further introduce MolmoSpaces-Bench, a benchmark suite spanning base skills and LLM-assisted long-horizon tasks, with controlled difficulty and perturbations that enable detailed analysis and principled comparison of generalist robot policies. We validate MolmoSpaces-Bench’s utility with thorough experimentation, demonstrating strong sim-to-real correlation and benchmarking multiple SOTA policies, including distributional evaluations that reveal subtle policy behavior characteristics. Although simulation remains inherently imperfect, well-designed simulation benchmarks provide a critical foundation for evaluating robotic policies and guiding progress toward robust real-world performance. In the future, we plan to support data generation and reinforcement learning for robot policies at scale, enabling us to further study scaling behaviors for robot foundation models.

###### Author Contributions

This project was made possible through the equal contributions of all four co-first authors, in no particular order. Their individual contributions are as follows:

- • Yejin Kim: Led the project and initially converted all assets and scenes from AI2-THOR to MuJoCo MJCF format; contributed to asset quality testings, grasp generation pipeline and evaluations, benchmark creation, and baseline evaluations.
- • Wilbert Pumacay: Led the asset conversion and quality testings across MuJoCo, Isaac and ManiSkill simulators.
- • Omar Rayyan: Led distributional benchmark evaluations, the addition of π, CAPs baselines and teleoperation data-collection; led the grasp generation pipeline and evaluations; contributed to benchmark creation.
- • Max Argus: Led the benchmark creation and baseline evaluations.

All other contributors are also deeply appreciated for their effort, which is critical to the success of the MolmoSpaces project. As not all of these can be captured, we indicate their primary contributing role in MolmoSpaces:

- • For assets conversion and physical parameter tunings: Yejin Kim, Wilbert Pumacay, Winson Han, Eli VanderBilt, Jordi Salvador, and Shuo Liu.
- • For grasp generation and evalautions: Omar Rayyan, and Yejin Kim.
- • For benchmark creation and baseline evaluation: Max Argus, Omar Rayyan, Yejin Kim, Arjun Guru, Maya Guru, Abhay Deshpande, Rose Hendrix, Snehal Jauhri, Shuo Liu, Ainaz Eftekhar, Ying-Chun Lee, and Piper Wolters. Nur Muhammad Mahi Shafiullah advised the zero-shot and distributional evaluations.
- • For multi-simulator support: Wilbert Pumacay, Alvaro Herrasti, and Donovan Clay
- • For paper writing and figures: Yejin Kim, Max Argus, Wilbert Pumacay, Omar Rayyan, Winson Han, Eli VanderBilt, Abhay Deshpande, Rose Hendrix, Maya Guru, Arjun Guru, Jordi Salvador, Jiafei Duan, Nur Muhammad Mahi Shafiullah, and Ranjay Krishna.
- • For project management: Karen Farley.
- • For research advisory: Ranjay Krishna, Dieter Fox, and Ali Farhadi.
- • Project PI: Ranjay Krishna

###### Acknowledgment

This work would not be possible without the support of our colleagues at Ai2:

- • We thank Rachel Ratner and Karen Goodfellow for the creation and the support of robot demo webpage for MolmoSpaces.
- • We thank David Albright, Crystal Nam, Kristin Cha, Sophie Lebrecht, Taira Anderson, Kyle Wiggers, Kelsey MacMillan, Katie Morigi, and Megan Bartot for project management, support to robot room and publicity of MolmoSpaces
- • We thank Yoganand Chandrasekhar, Johann Dahm, Fangzhou Hu, and Caroline Wu for their work on the Ai2 cluster.

###### References

- [1] Kevin Black, Noah Brown, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, Lachy Groom, Karol Hausman, Brian Ichter, Szymon Jakubczak, Tim Jones, Liyiming Ke, Sergey Levine, Adrian Li-Bell, Mohith Mothukuri, Suraj Nair, Karl Pertsch, Lucy Xiaoyang Shi, James Tanner, Quan Vuong, Anna Walling, Haohuan Wang, and Ury Zhilinsky. π0: A vision-language-action flow model for general robot control, 2026. URL https://arxiv.org/abs/2410.24164.
- [2] Physical Intelligence, Kevin Black, Noah Brown, James Darpinian, Karan Dhabalia, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, Manuel Y. Galliker, Dibya Ghosh, Lachy Groom, Karol Hausman, Brian Ichter, Szymon Jakubczak, Tim Jones, Liyiming Ke, Devin LeBlanc, Sergey Levine, Adrian Li-Bell, Mohith Mothukuri, Suraj Nair, Karl Pertsch, Allen Z. Ren, Lucy Xiaoyang Shi, Laura Smith, Jost Tobias Springenberg, Kyle Stachowicz, James Tanner, Quan Vuong, Homer Walke, Anna Walling, Haohuan Wang, Lili Yu, and Ury Zhilinsky. π0.5: a vision-language-action model with open-world generalization, 2025. URL https://arxiv.org/abs/2504.16054.
- [3] Gemini Robotics Team, Saminda Abeyruwan, Joshua Ainslie, Jean-Baptiste Alayrac, Montserrat Gonzalez Arenas, Travis Armstrong, Ashwin Balakrishna, Robert Baruch, Maria Bauza, Michiel Blokzijl, et al. Gemini robotics: Bringing ai into the physical world. arXiv preprint arXiv:2503.20020, 2025.
- [4] Johan Bjorck, Fernando Castañeda, Nikita Cherniadev, Xingye Da, Runyu Ding, Linxi Fan, Yu Fang, Dieter Fox, Fengyuan Hu, Spencer Huang, et al. Gr00t n1: An open foundation model for generalist humanoid robots. arXiv preprint arXiv:2503.14734, 2025.
- [5] Martin Sedlacek, Pavlo Yefanov, Georgy Ponimatkin, Jai Bardhan, Simon Pilc, Mederic Fourmy, Evangelos Kazakos, Cees GM Snoek, Josef Sivic, and Vladimir Petrik. Realm: A real-to-sim validated benchmark for generalization in robotic manipulation. arXiv preprint arXiv:2512.19562, 2025.
- [6] Bo Liu, Yifeng Zhu, Chongkai Gao, Yihao Feng, Qiang Liu, Yuke Zhu, and Peter Stone. Libero: Benchmarking knowledge transfer for lifelong robot learning. Advances in Neural Information Processing Systems, 36:44776–44791, 2023.
- [7] Oier Mees, Lukas Hermann, Erick Rosete-Beas, and Wolfram Burgard. Calvin: A benchmark for language-conditioned policy learning for long-horizon robot manipulation tasks, 2022. URL https://arxiv.org/abs/2112.03227.
- [8] Stephen James, Zicong Ma, David Rovick Arrojo, and Andrew J. Davison. Rlbench: The robot learning benchmark & learning environment, 2019. URL https://arxiv.org/abs/1909.12271.
- [9] Senyu Fei, Siyin Wang, Junhao Shi, Zihao Dai, Jikun Cai, Pengfang Qian, Li Ji, Xinzhe He, Shiduo Zhang, Zhaoye Fei, Jinlan Fu, Jingjing Gong, and Xipeng Qiu. Libero-plus: In-depth robustness analysis of vision-language-action models. CoRR, abs/2510.13626, 2025. doi: 10.48550/ARXIV.2510.13626. URL https://doi.org/10.48550/arXiv.2510.13626.
- [10] Xueyang Zhou, Yangming Xu, Guiyao Tie, Yongchao Chen, Guowen Zhang, Duanfeng Chu, Pan Zhou, and Lichao Sun. Libero-pro: Towards robust and fair evaluation of vision-language-action models beyond memorization, 2025. URL https: //arxiv.org/abs/2510.03827.
- [11] Anonymous. Robotarena infinity: Unlimited robot benchmarking via real-to-sim translation. In The Fourteenth International Conference on Learning Representations, 2026. URL https://openreview.net/forum?id=OutljIofvS.
- [12] Arhan Jain, Mingtong Zhang, Kanav Arora, William Chen, Marcel Torne, Muhammad Zubair Irshad, Sergey Zakharov, Yue Wang, Sergey Levine, Chelsea Finn, Wei Ma, Dhruv Shah, Abhishek Gupta, and Karl Pertsch. Polaris: Scalable real-to-sim evaluations for generalist robot policies, 2025.
- [13] Emanuel Todorov, Tom Erez, and Yuval Tassa. Mujoco: A physics engine for model-based control. 2012 IEEE/RSJ International Conference on Intelligent Robots and Systems, pages 5026–5033, 2012.
- [14] NVIDIA. Isaac Sim, 2026. URL https://github.com/isaac-sim/IsaacSim.
- [15] Tongzhou Mu, Z. Ling, Fanbo Xiang, Derek Yang, Xuanlin Li, Stone Tao, Zhiao Huang, Zhiwei Jia, and Hao Su. Maniskill: Generalizable manipulation skill benchmark with large-scale demonstrations. In NeurIPS Datasets and Benchmarks, 2021.
- [16] Karl Pertsch, Kyle Stachowicz, Brian Ichter, Danny Driess, Suraj Nair, Quan Vuong, Oier Mees, Chelsea Finn, and Sergey Levine. Fast: Efficient action tokenization for vision-language-action models. arXiv preprint arXiv:2501.09747, 2025.
- [17] Chenghao Yin, Da Huang, Di Yang, Jichao Wang, Nanshu Zhao, Chen Xu, Wenjun Sun, Linjie Hou, Zhijun Li, Junhui Wu, Zhaobo Liu, Zhenfei Xiao, Shenglan Zhang, Lei Bao, Rui Feng, Zhenquan Pang, Jiayu Li, Qian Wang, and Maoqing Yao. Genie sim 3.0 : A high-fidelity comprehensive simulation platform for humanoid robot, 2026.
- [18] Eric Kolve, Roozbeh Mottaghi, Daniel Gordon, Yuke Zhu, Abhinav Gupta, and Ali Farhadi. AI2-THOR: an interactive 3d environment for visual AI. CoRR, abs/1712.05474, 2017. URL http://arxiv.org/abs/1712.05474.

- [19] Manolis Savva, Abhishek Kadian, Oleksandr Maksymets, Yili Zhao, Erik Wijmans, Bhavana Jain, Julian Straub, Jia Liu, Vladlen Koltun, Jitendra Malik, et al. Habitat: A platform for embodied ai research. In Proceedings of the IEEE/CVF international conference on computer vision, pages 9339–9347, 2019.
- [20] Andrew Szot, Alex Clegg, Eric Undersander, Erik Wijmans, Yili Zhao, John Turner, Noah Maestre, Mustafa Mukadam, Devendra Chaplot, Oleksandr Maksymets, Aaron Gokaslan, Vladimir Vondrus, Sameer Dharur, Franziska Meier, Wojciech Galuba, Angel Chang, Zsolt Kira, Vladlen Koltun, Jitendra Malik, Manolis Savva, and Dhruv Batra. Habitat 2.0: Training home assistants to rearrange their habitat, 2022. URL https://arxiv.org/abs/2106.14405.
- [21] Matt Deitke, Eli VanderBilt, Alvaro Herrasti, Luca Weihs, Kiana Ehsani, Jordi Salvador, Winson Han, Eric Kolve, Aniruddha Kembhavi, and Roozbeh Mottaghi. Procthor: Large-scale embodied AI using procedural generation. In Sanmi Koyejo, S. Mohamed, A. Agarwal, Danielle Belgrave, K. Cho, and A. Oh, editors, Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 - December 9, 2022, 2022. URL http://papers.nips.cc/paper_files/paper/2022/hash/ 27c546ab1e4f1d7d638e6a8dfbad9a07-Abstract-Conference.html.
- [22] Soroush Nasiriany, Abhiram Maddukuri, Lance Zhang, Adeet Parikh, Aaron Lo, Abhishek Joshi, Ajay Mandlekar, and Yuke Zhu. Robocasa: Large-scale simulation of everyday tasks for generalist robots. In Robotics: Science and Systems, 2024.
- [23] Anonymous. Robocasa365: A large-scale simulation framework for training and benchmarking generalist robots. In The Fourteenth International Conference on Learning Representations, 2026. URL https://openreview.net/forum?id=tQJYKwc3n4.
- [24] Abby O’Neill, Abdul Rehman, Abhiram Maddukuri, Abhishek Gupta, Abhishek Padalkar, Abraham Lee, Acorn Pooley, Agrim Gupta, Ajay Mandlekar, Ajinkya Jain, et al. Open x-embodiment: Robotic learning datasets and rt-x models: Open x-embodiment collaboration 0. In 2024 IEEE International Conference on Robotics and Automation (ICRA), pages 6892–6903. IEEE, 2024.
- [25] Matt Deitke, Dustin Schwenk, Jordi Salvador, Luca Weihs, Oscar Michel, Eli VanderBilt, Ludwig Schmidt, Kiana Ehsani, Aniruddha Kembhavi, and Ali Farhadi. Objaverse: A universe of annotated 3d objects. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2023, Vancouver, BC, Canada, June 17-24, 2023, pages 13142–13153. IEEE, 2023. doi: 10.1109/CVPR52729.2023.01263. URL https://doi.org/10.1109/CVPR52729.2023.01263.
- [26] Yue Yang, Fan-Yun Sun, Luca Weihs, Eli VanderBilt, Alvaro Herrasti, Winson Han, Jiajun Wu, Nick Haber, Ranjay Krishna, Lingjie Liu, Chris Callison-Burch, Mark Yatskar, Aniruddha Kembhavi, and Christopher Clark. Holodeck: Language guided generation of 3d embodied AI environments. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2024, Seattle, WA, USA, June 16-22, 2024, pages 16277–16287. IEEE, 2024. doi: 10.1109/CVPR52733.2024.01536. URL https://doi.org/10.1109/CVPR52733.2024.01536.
- [27] Weipeng Zhong, Peizhou Cao, Yichen Jin, Li Luo, Wenzhe Cai, Jingli Lin, Hanqing Wang, Zhaoyang Lyu, Tai Wang, Bo Dai, Xudong Xu, and Jiangmiao Pang. Internscenes: A large-scale simulatable indoor scene dataset with realistic layouts, 2025. URL https://arxiv.org/abs/2509.10813.
- [28] Adithyavairavan Murali, Balakumar Sundaralingam, Yu-Wei Chao, Wentao Yuan, Jun Yamada, Mark T. Carlson, Fabio Ramos, Stanley T. Birchfield, Dieter Fox, and Clemens Eppner. Graspgen: A diffusion-based framework for 6-dof grasping with on-generator training. ArXiv, abs/2507.13097, 2025.
- [29] Pranav Atreya, Karl Pertsch, Tony Lee, Moo Jin Kim, Arhan Jain, Artur Kuramshin, Clemens Eppner, Cyrus Neary, Edward Hu, Fabio Ramos, Jonathan Tremblay, Kanav Arora, Kirsty Ellis, Luca Macesanu, Matthew Leonard, Meedeum Cho, Ozgur Aslan, Shivin Dass, Jie Wang, Xingfang Yuan, Xuning Yang, Abhishek Gupta, Dinesh Jayaraman, Glen Berseth, Kostas Daniilidis, Roberto Martin Martin, Youngwoon Lee, Percy Liang, Chelsea Finn, and Sergey Levine. Roboarena: Distributed real-world evaluation of generalist robot policies. ArXiv, abs/2506.18123, 2025.
- [30] Zhiyuan Zhou, Pranav Atreya, You Liang Tan, Karl Pertsch, and Sergey Levine. Autoeval: Autonomous evaluation of generalist robot manipulation policies in the real world. arXiv preprint arXiv:2503.24278, 2025.
- [31] Mohit Shridhar, Jesse Thomason, Daniel Gordon, Yonatan Bisk, Winson Han, Roozbeh Mottaghi, Luke Zettlemoyer, and Dieter Fox. Alfred: A benchmark for interpreting grounded instructions for everyday tasks, 2020. URL https://arxiv.org/abs/ 1912.01734.
- [32] Karmesh Yadav, Jacob Krantz, Ram Ramrakhya, Santhosh Kumar Ramakrishnan, Jimmy Yang, Austin Wang, John Turner, Aaron Gokaslan, Vincent-Pierre Berges, Roozbeh Mootaghi, Oleksandr Maksymets, Angel X Chang, Manolis Savva, Alexander Clegg, Devendra Singh Chaplot, and Dhruv Batra. Habitat challenge 2023. https://aihabitat.org/challenge/2023/, 2023.
- [33] Haoran Geng, Feishi Wang, Songlin Wei, Yuyang Li, Bangjun Wang, Boshi An, Charlie Tianyue Cheng, Haozhe Lou, Peihao Li, Yen-Jen Wang, Yutong Liang, Dylan Goetting, Chaoyi Xu, Haozhe Chen, Yuxi Qian, Yiran Geng, Jiageng Mao, Weikang Wan,

- Mingtong Zhang, Jiangran Lyu, Siheng Zhao, Jiazhao Zhang, Jialiang Zhang, Chengyang Zhao, Haoran Lu, Yufei Ding, Ran Gong, Yuran Wang, Yuxuan Kuang, Ruihai Wu, Baoxiong Jia, Carlo Sferrazza, Hao Dong, Siyuan Huang, Yue Wang, Jitendra Malik, and Pieter Abbeel. Roboverse: Towards a unified platform, dataset and benchmark for scalable and generalizable robot learning. ArXiv, abs/2504.18904, 2025.
- [34] Shiduo Zhang, Zhe Xu, Peiju Liu, Xiaopeng Yu, Yuan Li, Qinghui Gao, Zhaoye Fei, Zhangyue Yin, Zuxuan Wu, Yu-Gang Jiang, and Xipeng Qiu. Vlabench: A large-scale benchmark for language-conditioned robotics manipulation with long-horizon reasoning tasks. CoRR, abs/2412.18194, 2024. doi: 10.48550/ARXIV.2412.18194. URL https://doi.org/10.48550/arXiv. 2412.18194.
- [35] Wilbert Pumacay, Ishika Singh, Jiafei Duan, Ranjay Krishna, Jesse Thomason, and Dieter Fox. The colosseum: A benchmark for evaluating generalization for robotic manipulation. arXiv preprint arXiv:2402.08191, 2024.
- [36] Zhijie Wang, Zhehua Zhou, Jiayang Song, Yuheng Huang, Zhan Shu, and Lei Ma. Vlatest: Testing and evaluating visionlanguage-action models for robotic manipulation. Proc. ACM Softw. Eng., 2(FSE), jul 2025. doi: 10.1145/3729343. URL https://doi.org/10.1145/3729343.
- [37] Chengshu Li, Ruohan Zhang, Josiah Wong, Cem Gokmen, Sanjana Srivastava, Roberto Martín-Martín, Chen Wang, Gabrael Levine, Michael Lingelbach, Jiankai Sun, Mona Anvari, Minjune Hwang, Manasi Sharma, Arman Aydin, Dhruva Bansal, Samuel Hunter, Kyu-Young Kim, Alan Lou, Caleb R Matthews, Ivan Villa-Renteria, Jerry Huayang Tang, Claire Tang, Fei Xia, Silvio Savarese, Hyowon Gweon, Karen Liu, Jiajun Wu, and Li Fei-Fei. BEHAVIOR-1k: A benchmark for embodied AI with 1,000 everyday activities and realistic simulation. In 6th Annual Conference on Robot Learning, 2022. URL https://openreview.net/forum?id=_8DoIe8G3t.
- [38] Arth Shukla, Stone Tao, and Hao Su. Maniskill-hab: A benchmark for low-level manipulation in home rearrangement tasks. ArXiv, abs/2412.13211, 2024.
- [39] Rui Yang, Hanyang Chen, Junyu Zhang, Mark Zhao, Cheng Qian, Kangrui Wang, Qineng Wang, Teja Venkat Koripella, Marziyeh Movahedi, Manling Li, Heng Ji, Huan Zhang, and Tong Zhang. Embodiedbench: Comprehensive benchmarking multi-modal large language models for vision-driven embodied agents, 2025. URL https://arxiv.org/abs/2502.09560.
- [40] Xuanlin Li, Kyle Hsu, Jiayuan Gu, Karl Pertsch, Oier Mees, Homer Rich Walke, Chuyuan Fu, Ishikaa Lunawat, Isabel Sieh, Sean Kirmani, Sergey Levine, Jiajun Wu, Chelsea Finn, Hao Su, Quan Vuong, and Ted Xiao. Evaluating real-world robot manipulation policies in simulation. arXiv preprint arXiv:2405.05941, 2024.
- [41] Jianxiong Xiao, James Hays, Krista A. Ehinger, Aude Oliva, and Antonio Torralba. SUN database: Large-scale scene recognition from abbey to zoo. In The Twenty-Third IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2010, San Francisco, CA, USA, 13-18 June 2010, pages 3485–3492. IEEE Computer Society, 2010. doi: 10.1109/CVPR.2010.5539970. URL https://doi.org/10.1109/CVPR.2010.5539970.
- [42] Tao Ge, Xin Chan, Xiaoyang Wang, Dian Yu, Haitao Mi, and Dong Yu. Scaling synthetic data creation with 1,000,000,000 personas. arXiv preprint arXiv:2406.20094, 2024.
- [43] Xinyue Wei, Minghua Liu, Zhan Ling, and Hao Su. Approximate convex decomposition for 3d meshes with collision-aware concavity and tree search. ACM Transactions on Graphics (TOG), 41(4):1–18, 2022.
- [44] The Allen Institue for AI. objathor, 2024. URL https://github.com/allenai/objathor. GitHub repository.
- [45] John P. McCrae, Alexandre Rademaker, Francis Bond, Ewa Rudnicka, and Christiane Fellbaum. English wordnet 2019 - an open-source wordnet for english. In Piek Vossen and Christiane Fellbaum, editors, Proceedings of the 10th Global Wordnet Conference, GWC 2019, Wroclaw, Poland, July 23-27, 2019, pages 245–252. Global Wordnet Association, 2019. URL https://aclanthology.org/2019.gwc-1.31/.
- [46] John P. McCrae and Alexandre Rademaker and Ewa Rudnicka and Bernard Bou and Daiki Nomura and David Cillessen and Ciara O’Loughlin and Cathal McGovern and Francis Bond and Eric Kafe and Michael Wayne Goodman and Merrick Choo Yeu Herng and Enejda Nasaj. Open English Wordnet, 2022. URL https://github.com/globalwordnet/english-wordnet/ releases/tag/2022-edition. GitHub repository.
- [47] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.
- [48] Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. GPT-4o system card. arXiv preprint arXiv:2410.21276, 2024.
- [49] OpenAI. GPT-4.1, 2024. URL https://platform.openai.com/. Proprietary large language model accessed via the OpenAI API; no public system card available.

- [50] Arsalan Mousavian, Clemens Eppner, and Dieter Fox. 6-dof graspnet: Variational grasp generation for object manipulation. In International Conference on Computer Vision (ICCV), 2019.
- [51] Clemens Eppner, Arsalan Mousavian, and Dieter Fox. Acronym: A large-scale grasp dataset based on simulation. 2021 IEEE International Conference on Robotics and Automation (ICRA), pages 6222–6227, 2020.
- [52] Zichen Jeff Cui, Omar Rayyan, Haritheja Etukuru, Bowen Tan, Zavier Andrianarivo, Zicheng Teng, Yihang Zhou, Krish Mehta, Nicholas Wojno, Kevin Yuanbo Wu, Manan H Anjaria, Ziyuan Wu, Manrong Mao, Guangxun Zhang, Binit Shah, Yejin Kim, Soumith Chintala, Lerrel Pinto, and Nur Muhammad Mahi Shafiullah. Contact-anchored policies: Contact conditioning creates strong robot utility models. arXiv preprint arXiv:2602.09017, 2026.
- [53] Alexander Khazatsky, Karl Pertsch, Suraj Nair, Ashwin Balakrishna, Sudeep Dasari, Siddharth Karamcheti, Soroush Nasiriany, Mohan Kumar Srirama, Lawrence Yunliang Chen, Kirsty Ellis, Peter David Fagan, Joey Hejna, Masha Itkina, Marion Lepert, Yecheng Jason Ma, Patrick Tree Miller, Jimmy Wu, Suneel Belkhale, Shivin Dass, Huy Ha, Arhan Jain, Abraham Lee, Youngwoon Lee, Marius Memmel, Sungjae Park, Ilija Radosavovic, Kaiyuan Wang, Albert Zhan, Kevin Black, Cheng Chi, Kyle Beltran Hatch, Shan Lin, Jingpei Lu, Jean Mercat, Abdul Rehman, Pannag R Sanketi, Archit Sharma, Cody Simpson, Quan Vuong, Homer Rich Walke, Blake Wulfe, Ted Xiao, Jonathan Heewon Yang, Arefeh Yavary, Tony Z. Zhao, Christopher Agia, Rohan Baijal, Mateo Guaman Castro, Daphne Chen, Qiuyu Chen, Trinity Chung, Jaimyn Drake, Ethan Paul Foster, Jensen Gao, Vitor Guizilini, David Antonio Herrera, Minho Heo, Kyle Hsu, Jiaheng Hu, Muhammad Zubair Irshad, Donovon Jackson, Charlotte Le, Yunshuang Li, Kevin Lin, Roy Lin, Zehan Ma, Abhiram Maddukuri, Suvir Mirchandani, Daniel Morton, Tony Nguyen, Abigail O’Neill, Rosario Scalise, Derick Seale, Victor Son, Stephen Tian, Emi Tran, Andrew E. Wang, Yilin Wu, Annie Xie, Jingyun Yang, Patrick Yin, Yunchu Zhang, Osbert Bastani, Glen Berseth, Jeannette Bohg, Ken Goldberg, Abhinav Gupta, Abhishek Gupta, Dinesh Jayaraman, Joseph J Lim, Jitendra Malik, Roberto Martín-Martín, Subramanian Ramamoorthy, Dorsa Sadigh, Shuran Song, Jiajun Wu, Michael C. Yip, Yuke Zhu, Thomas Kollar, Sergey Levine, and Chelsea Finn. Droid: A large-scale in-the-wild robot manipulation dataset, 2024.
- [54] Kevin Zakka, Yuval Tassa, and MuJoCo Menagerie Contributors. MuJoCo Menagerie: A collection of high-quality simulation models for MuJoCo, 2022. URL http://github.com/google-deepmind/mujoco_menagerie.
- [55] Balakumar Sundaralingam, Siva Kumar Sastry Hari, Adam Fishman, Caelan Reed Garrett, Karl Van Wyk, Valts Blukis, Alexander Millane, Helen Oleynikova, Ankur Handa, Fabio Tozeto Ramos, Nathan D. Ratliff, and Dieter Fox. Curobo: Parallelized collision-free robot motion generation. 2023 IEEE International Conference on Robotics and Automation (ICRA), pages 8112–8119, 2023.
- [56] Omar Rayyan, Maximilian Gilles, and Yuchen Cui. Teledex: Accessible dexterous teleoperation, 2026. URL https: //github.com/omarrayyann/teledex.
- [57] Kiana Ehsani, Tanmay Gupta, Rose Hendrix, Jordi Salvador, Luca Weihs, Kuo-Hao Zeng, Kunal Pratap Singh, Yejin Kim, Winson Han, Alvaro Herrasti, Ranjay Krishna, Dustin Schwenk, Eli VanderBilt, and Aniruddha Kembhavi. Spoc: Imitating shortest paths in simulation enables effective navigation and manipulation in the real world. 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 16238–16250, 2023. URL https://api.semanticscholar.org/ CorpusID:271769346.
- [58] Gemini Team, Petko Georgiev, Ving Ian Lei, Ryan Burnell, Libin Bai, Anmol Gulati, Garrett Tanzer, Damien Vincent, Zhufeng Pan, Shibo Wang, et al. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. arXiv preprint arXiv:2403.05530, 2024.
- [59] Ainaz Eftekhar, Luca Weihs, Rose Hendrix, Ege Caglar, Jordi Salvador, Alvaro Herrasti, Winson Han, Eli VanderBil, Aniruddha Kembhavi, Ali Farhadi, Ranjay Krishna, Kiana Ehsani, and Kuo-Hao Zeng. The one ring: a robotic indoor navigation generalist. ArXiv, abs/2412.14401, 2024.
- [60] Meng Wei, Chenyang Wan, Jiaqi Peng, Xiqian Yu, Yuqiang Yang, Delin Feng, Wenzhe Cai, Chenming Zhu, Tai Wang, Jiangmiao Pang, and Xihui Liu. Ground slow, move fast: A dual-system foundation model for generalizable vision-and-language navigation. ArXiv, abs/2512.08186, 2025. URL https://api.semanticscholar.org/CorpusID:283711459.
- [61] Chendi Lin, Heshan Liu, Qunshu Lin, Zachary Bright, Shitao Tang, Yihui He, Minghao Liu, Ling Zhu, and Cindy Le. Objaverse++: Curated 3D Object Dataset with Quality Annotations. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) Workshops, pages 6813–6822, October 2025.
- [62] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In ICML, 2021.
- [63] Gabriel Ilharco, Mitchell Wortsman, Ross Wightman, Cade Gordon, Nicholas Carlini, Rohan Taori, Achal Dave, Vaishaal

- Shankar, Hongseok Namkoong, John Miller, Hannaneh Hajishirzi, Ali Farhadi, and Ludwig Schmidt. Openclip, July 2021. URL https://doi.org/10.5281/zenodo.5143773.
- [64] Mehdi Cherti, Romain Beaumont, Ross Wightman, Mitchell Wortsman, Gabriel Ilharco, Cade Gordon, Christoph Schuhmann, Ludwig Schmidt, and Jenia Jitsev. Reproducible scaling laws for contrastive language-image learning. In CVPR, 2023.

###### Appendix A Object Model Dataset Details

###### patient exam room

procedure room

waiting area reception area

nurse station

post-anesthesia care unit (pacu)

hallway

staff administration

corridor

treatment room

restroom

adoption counseling room

animal quarantine room

consultation room

animal exam room

staff office

chemotherapy room

utility room storage room

pet play room

sterilization room

infusion room

private counseling room

animal assessment room

phlebotomy room medical records room

medication preparation room

triage room grooming room

immunization room

animal intake room

child observation room panoramic imaging room

dental x-ray room

###### rock.n.01

isolation room

relaxation room procedure surgery room feedingroom lab sample collection room minor surgery room phlebotomy sample collection room

media home theater

shared kitchen pantry coworking room yogaroom

scrub room pet bathing room

family consultation room

media room theater meditation room

diagnostic imaging room

equipment storage room

mezzanine sleeping area

pre-op room

parent consultation room

imaging room

dental laboratory family therapy room

pet meet-and-greet room

psychological testing room

animal isolation room

lactation room

x-rayroom surgery room

treatment bay vaccination room

sauna room

kid'snook media room sparoom

group therapy room

library room

baggage room study office

linenroom

janitor room

guestroom activity room craftroom

therapy room breakfast area gameroom

baggage storage meeting room

kids' playroom staffroom

biopsyroom counseling room

observation room

playroom pet isolation room quietroom

breakfast room recreation room study alcove

patient lounge

###### classroom

global_positioning_system.n.01

personal_digital_assistant.n.01

point-and-shoot_camera.n.01

videocassette_recorder.n.01

electromechanical_device.n.01

medical exam room musicroom hobbyroom privategym small reception petroom winecellar

motion-picture_camera.n.01

games lounge studyroom smallden

mythical_monster.n.01

computer_accessory.n.01

cassette_deck.n.01 doorbell.n.01 dry_battery.n.01 terminal.n.02 memory_device.n.01 resistor.n.01 trackball.n.01 radio-phonograph.n.01 minicomputer.n.01 turntable.n.01 screen.n.04 screen.n.01 interphone.n.01 television_antenna.n.01 microprocessor.n.01 token.n.03 cassette_tape.n.01 memory.n.04

personal_computer.n.01

optical_instrument.n.01

portable_computer.n.01

preproom waitingroom

entertainment_center.n.01

desk_phone.n.01 push-button_radio.n.01 disk_drive.n.01

recording_system.n.01

flashlight_battery.n.01 toggle_switch.n.01

cassette_recorder.n.01

skull.n.01 bust.n.03

computer_monitor.n.01

artstudio

digital_voltmeter.n.01 keyboard.n.01

removable_disk.n.01

hand_calculator.n.01 clock_radio.n.01

electrical_device.n.01 videocassette.n.01

network_bridge.n.01 cpu_board.n.01

desktop_computer.n.01

cellular_telephone.n.01

storage locker room librarynook computer room tvlounge workshop

digital_display.n.01

electronic_device.n.01

pay-phone.n.01 electrical_cable.n.01 diskette.n.01

hand-held_computer.n.01

external_drive.n.01 switch.n.01

optical_device.n.01

audio_amplifier.n.01

adapter.n.02 flash_camera.n.01 model_t.n.01 car_battery.n.01 handset.n.01

record_player.n.01

tape_recorder.n.01 power_pack.n.01

gramophone.n.01 memory_chip.n.01 model.n.07

printed_circuit.n.01 cassette_player.n.01 receiver.n.01

space_probe.n.01

polaroid_camera.n.01

flash_memory.n.01

effects_unit.n.01

oscilloscope.n.01

music practice room conference room

camera_lens.n.01

reflex_camera.n.01

box_camera.n.01 camcorder.n.01 ipod.n.01

ghetto_blaster.n.01 device.n.04

digital_clock.n.01

transformer.n.01

cd_player.n.01 servo.n.01

push_button.n.01

connection.n.03 module.n.03

digital_camera.n.01 console.n.02

readingnook homegym

heat_sink.n.01

gateway.n.01

circuit_board.n.01

computer.n.01

android.n.01

telephone.n.01 casque.n.01

solar_array.n.01

monitor.n.06

console.n.03

webcam.n.01 cable.n.02

computer_keyboard.n.01

screen.n.03 ipad.n.01

mouse.n.01

viewer.n.02

antenna.n.01 stylus.n.02

charger.n.02

robotics_equipment.n.01

luggagestorage lobby pantry mudroom walk-incloset readingroom communal lounge

radar.n.01

remote_control.n.01 mouse.n.04

generator.n.02

iphone.n.01

alarm_clock.n.01 terminal.n.04

router.n.02

dial_telephone.n.01

control.n.09

matrix.n.02

plug.n.05

earphone.n.01

walkie-talkie.n.01

battery.n.02 can.n.01

joystick.n.02

control_panel.n.01

laptop.n.01

satellite.n.01 tablet.n.05

compact.n.03

radio_receiver.n.01

mobile.n.03

disk.n.02

camera.n.01

television_receiver.n.01

screen.n.08

rendering.n.06

headset.n.01

smart_phone.n.01

tablet.n.01

miniature.n.02

disk.n.01

studylounge

communalkitchen sharedlounge

device.n.01

seashell.n.01

sharedkitchenette gamesroom

utilityroom

reception

homeoffice

european_dragon.n.01

entrance

fossil.n.02

library

death's_head.n.01

model.n.04

eolith.n.01 paleolith.n.01

objet_d'art.n.01

laundryroom foyer

ammonite.n.01

###### sword.n.01

potsherd.n.01

bronze.n.02 tyrannosaur.n.01

replica.n.01

moss_agate.n.01

unicorn.n.01 triceratops.n.01

chambered_nautilus.n.01

trophy.n.02 artifact.n.01

gryphon.n.01 sarcophagus.n.01

golliwog.n.01 taxidermy.n.01

mineral.n.01

memorial.n.03 petroglyph.n.01

armored_dinosaur.n.01

heraldry.n.02 moon_shell.n.01

grimoire.n.01

ivory.n.01 antiquity.n.03 jasper.n.01

obsidian.n.01 lapis_lazuli.n.01

sphinx.n.03 rhodonite.n.01

geode.n.01

storageroom

wyvern.n.01 egyptian_deity.n.01 hadrosaur.n.01

tessera.n.01 actinolite.n.01 sand_dollar.n.01

leprechaun.n.01

love-token.n.01

dinosaur.n.01

pterosaur.n.01 ornithomimid.n.01

antique.n.02 verd_antique.n.01

plesiosaur.n.01

carnosaur.n.01

trading_card.n.01

collectible.n.01 tiger_cowrie.n.01

chalcedony.n.01

ghoul.n.02 malachite.n.01 pyrite.n.01

knight.n.01

crocodile.n.01

moonstone.n.01

trophy.n.01

bare_bone.n.01

rock.n.02 trilobite.n.01

americana.n.01

meteorite.n.01

golem.n.01

penguin.n.01

award.n.02

snuffbox.n.01

diamond.n.01

chimera.n.02

jack.n.06

troll.n.01

restroom corridor

###### rifle.n.01

chinese_puzzle.n.01 bishop.n.03 playing_card.n.01 puppet.n.03

board_game.n.01 yo-yo.n.01 spinner.n.02 kaleidoscope.n.02 cigar.n.01

jigsaw_puzzle.n.01 puppet.n.01

dartboard.n.01

rawhide.n.01 slide.n.04

domino.n.04

building_block.n.02

pistol.n.01 battle-ax.n.01

dollhouse.n.01 frisbee.n.01

pinata.n.01

bunny.n.02 ball.n.01

domino.n.03 dart.n.01 tenpin.n.01

jack-in-the-box.n.01 beach_ball.n.01

pinwheel.n.03

playhouse.n.01 kite.n.03 slingshot.n.01

choo-choo.n.01

meccano.n.01 goblin.n.01

pawn.n.03

ogre.n.02

chessman.n.01

doodlebug.n.01

hallway

water_pistol.n.01

cockhorse.n.01

fairy.n.01

rag_doll.n.01

snowman.n.01

teddy.n.01

puzzle.n.02

arrowhead.n.01 assault_rifle.n.01

lego.n.01

dollhouse.n.02

ball.n.05

urgent care center

sniper_rifle.n.01

outpatient clinic mental health center

healthcare facility

toy.n.03

diningarea

revolver.n.01 shotgun.n.01

veterinary clinic

popgun.n.01

doll.n.01

surgical center

hallway

oncology clinic

animal shelter pet boarding center

administrativeoffice

firearm.n.01 dagger.n.01

pediatric clinic

sphere.n.02

medical clinic

hand_grenade.n.01 submachine_gun.n.01

toy_soldier.n.01

kitchen

senior living facility smallhotel

dental clinic

semiautomatic_pistol.n.01

maul.n.01 cannon.n.01

hospital wing

gun.n.01

dummy.n.03

broadsword.n.01 crossbow.n.01

weapon.n.01 fighter.n.02

mace.n.04 flintlock.n.01

automaton.n.02

armet.n.01 poleax.n.02

artschool culinaryschool

cavalry_sword.n.01

preschoolordaycare languagelearningcenter

lobby

cork_gun.n.01 launcher.n.01

spear.n.01 machine_gun.n.01 kalashnikov.n.01 sawed-off_shotgun.n.01 tommy_gun.n.01 dragunov.n.01 hunting_knife.n.01 tomahawk.n.01 scimitar.n.01 hilt.n.01 bludgeon.n.01 armor.n.01 bullet.n.01 telescopic_sight.n.01 turret.n.01 gatling_gun.n.01 trident.n.01

entrance

penthouse

rocket.n.01 falchion.n.01

storageroom supplyroom

heaume.n.01

livingroom

full_metal_jacket.n.01

arrow.n.02 missile.n.01

target.n.04 grenade.n.01 cuirass.n.01 carbine.n.01 air_gun.n.01

spearhead.n.03

flail.n.01 bomb.n.01

tank.n.02 gunsight.n.01 winchester.n.02 catapult.n.03 smoke_bomb.n.01

knight.n.02

blunderbuss.n.01

bayonet.n.01 flamethrower.n.01 khukuri.n.01 trench_knife.n.01 halberd.n.01 luger.n.01 silencer.n.01 very_pistol.n.01

bow.n.08

ballista.n.01

maxim_gun.n.01

light_machine_gun.n.01

bazooka.n.01 fencing_sword.n.01

paintball_gun.n.01

handbow.n.01 coat_of_arms.n.01

arrow.n.01 longbow.n.01

pump_action.n.01

projectile.n.01 brass_knucks.n.01 quaker_gun.n.01 zip_gun.n.01

saber.n.01

mauser.n.02

trebuchet.n.01

scorpion.n.03 plate_armor.n.02 shackle.n.02 armor_plate.n.01 dynamite.n.01 shotgun_shell.n.01 switchblade.n.01 explosive_device.n.01 harpoon_gun.n.01 panzer.n.01 battering_ram.n.01 stun_gun.n.01 cockatrice.n.01

warrior.n.01

man-at-arms.n.01

roc.n.01 derringer.n.01

quiver.n.03 artillery_shell.n.01 rapier.n.01 breastplate.n.01 rocket.n.02 spike.n.11 land_mine.n.01 powder_horn.n.01 truncheon.n.01 scabbard.n.01 uzi.n.01

naval_gun.n.01

case_shot.n.01

rifleman.n.02

toy.n.02

colt.n.02

apartment

teacherlounge stafflounge

ensemblerehearsalroom

elementaryschool

guesthouse

###### sculpture.n.01

testkitchen smalldancestudio sciencelab

changingroom practiceroom readingnook

dormitory

musicschool highschool

drawingstudio resourceroom

Collectibles (8.0%)

sculpturestudio

indoorplayroom pastrykitchen

Electronics (8.0%)

instrumentstorage artstudio naproom

basementapartment loft

coldstorageroom photographydarkroom digitalmedialab ingredientpreproom printmakingroom musicpracticeroom propsstorage audio-visuallab

bathroom

WeaponsandArmory(7.4%)

examroom smallgrouptutoringroom counselingoffice conversationroom studyroom studentclubroom studycubicles careercounselingoffice sensoryplayroom recordingbooth smallseminarroom

danceacademy schoolbuilding

computerlab musictheoryclassroom smallgroupinterventionroom readingcorner

musiccontrolroom

readinglibrary foodplatingroom projectroom stretchingroom languagelab warm-uproom

mobilehome

###### statue.n.01

officearea

(19.1%) (12.8%)healthcare facility

plaything.n.01

middleschool

house

ArtInstallationsandExhibitDisplays(7.2%) DecorativeItems(5.3%)

###### relief.n.10 stele.n.02

ToysandGames(16.1%)

conferencevenue governmentoffice

deskarea

###### lay_figure.n.01

hostel

schoolbuilding

woodcarving.n.01

residentialbuilding

triumphal_arch.n.01

boardinghouse

(12.7%)

bedroom

restroom

###### figurine.n.01

co-livingspace

lawfirm officebuilding architecturefirm

hallway

residentialbuilding

officebuilding (12.6%)

studioorworkshop(5.7%)

musicrecordingstudio

WorkshopMachinery(0.0%)

PlaygroundEquipment(0.0%)

PetSuppliesandPetCareEquipment(0.0%)

StageandLightingEquipment(0.0%)

TextilesandFabrics(0.0%)

Stationery(0.1%)

UniformsandWorkwear(0.1%)

HospitalitySupplies(0.1%)

vase.n.01 decoration.n.01 emblem.n.01

MoneyandCurrency(0.1%)

BakingandBrewingEquipment(0.1%)

corridor

BeddingandLinens(0.1%)

electronicorappliancerepairworkshop

PackagingandShippingSupplies(0.2%)

OfficeEquipment(0.1%)

PersonalCareItems(0.2%)

HealthandMedical(0.2%)

productionkitchen

(8.5%)retail store recreationsportsfacility

CleaningSupplies(0.2%)

SoundandAudioEquipment(0.2%)

makeupanddressingroom

VisualandProjectionEquipment(0.2%)

LaboratoryEquipmentandGlassware(0.2%)

instrumentstorage

Plants(0.2%)

Writing,ArtandCraftToolsandSupplies(0.2%)

inspectionandqualitycontrolroom

textileworkshop

componentstorage

BuildingMaterials(5.2%)

toolsharpeningroom

BathroomItems(0.2%)

sewingroom

personal service center(8.5%)

SnacksandConfectionery(0.3%)

cuttingroom

GardeningToolsandSupplies(0.3%)

mixingroom

photoeditingsuite

filmtvproductionstudio

changingroom

HVACandClimateControl(0.4%)

meetingroom

propstorage

assemblyroom

finishingroom

MedicalEquipment(0.4%)

toolroom

diagnosticsroom

callcenter corporateoffice coworkingspace makerspace artgallery children'smuseum planetarium sciencemuseum museum naturalhistorymuseum exhibitioncenter artmuseum restaurant tastingroom fast food outlet print copy center barbershop bar beauty salon personal service center

partsstorage

weldingroom

museum (8.6%)

Footwear(0.4%)

materialsstorage

(5.8%)

carpentryshop

OfficeSupplies(0.5%)

precisionassemblyroom

liverecordingroom

woodstorageroom

photographystudio

Animals(0.5%)

metalworkingormachineshop

testingroom

###### loudspeaker.n.01

receptionarea

Accessories(0.5%)

ToolsandHardware(4.3%)

conferenceroom

Appliances(0.6%)

utilityroom

MusicalInstrumentsandEquipment(0.7%)

DisplayFixtures(0.6%)

SportsEquipmentandFitnessEquipment(1.1%)

EducationalMaterials(0.7%)

equipmentroom

(5.8%)

ProtectiveEnclosuresandCages(0.7%)

###### building.n.01

studioorworkshop

waitingarea

Beverageware(0.7%)

clockmakingworkshop

waitingarea receptionarea

administration

filter-tipped_cigarette.n.01

truss.n.02 stone.n.02 wall_panel.n.01 fieldstone.n.01 architecture.n.01 block.n.01

chocolate_candy.n.01

Storage Containers (3.1%) Lighting Fixtures (2.9%)

office

gingerbread_man.n.01

chocolate_chip_cookie.n.01

KitchenUtensils(0.8%)

Clothing Accessories (1.4%) Safety and Protective Gear (1.2%)

Jewelry(0.8%)

venus's_flytrap.n.01

fountain_grass.n.01

bulbous_plant.n.01

Vehicles(3.7%)

fisherman's_lure.n.01

garden_spade.n.01

pruning_shears.n.01

mainstudioarea

garden_trowel.n.01

planter.n.03

Tableware(0.9%)

evaporative_cooler.n.01

ventilation_shaft.n.01

wind_turbine.n.01

panel_heating.n.01

- air_conditioner.n.01

electric_heater.n.01

space_heater.n.01

anemometer.n.01

- air_conditioner.n.02

Kitchenware(1.0%)

bakery

OutdoorItems(1.1%)

Entertainment(1.0%)

electric_fan.n.01

suppliesroom

cheesemakingworkshop

sphygmomanometer.n.01

ultrasound_scanner.n.01

breakroom breakkitchenette

refracting_telescope.n.01

medical_instrument.n.01

space_lattice.n.01

hypodermic_syringe.n.01

stethoscope.n.01

geiger_counter.n.01

thermometer.n.01

skeletal_structure.n.01

Signage and Labels (1.6%)

Miscellaneous Items (1.8%)

Food and Drinks (1.6%)

brewery

storageroom

roller_skate.n.01

cowboy_boot.n.01

rubber_boot.n.01

high-heeled_shoe.n.01

walking_shoe.n.01

combat_boot.n.02

ankle_boot.n.01

column.n.06 panel.n.01

Furniture (2.4%)

running_shoe.n.01

storageroom serverroom

gym_shoe.n.01

productionkitchen

Home Decor (1.9%)

paper_clip.n.01 perpetual_calendar.n.01 corkboard.n.01 color_chart.n.01 rubber_band.n.01 attachment.n.02 name_tag.n.01 cartridge_holder.n.01 rubber_eraser.n.01 document.n.01 bulletin_board.n.02 bar_chart.n.01 newspaper.n.03 bulldog_clip.n.01

mainworkarea

pencil_sharpener.n.01

letter_opener.n.01

file_folder.n.01

writing_board.n.01

door.n.01 slab.n.01 cube.n.05 outcrop.n.01

handstamp.n.01

bookend.n.01

business_card.n.01

speech_balloon.n.01

thumbtack.n.01

notebook.n.01

calendar.n.03

standard.n.06

blackboard.n.01

waste_paper.n.01

cartridge.n.03

wastepaper_basket.n.01

projectassemblyarea

cartridge.n.01

seahorse.n.02 great_white_shark.n.01 dragonfly.n.01 labrador_retriever.n.01 african_elephant.n.01 true_frog.n.01 stag_beetle.n.01 domestic_cat.n.01 jellyfish.n.02 sea_turtle.n.01 sparrow.n.01 german_shepherd.n.01 tiger_moth.n.01 ant_lion.n.02

anglerfish.n.01

cowhide.n.02

grasshopper.n.01

praying_mantis.n.01

tarantula.n.02

pipefish.n.01

structure.n.01 gable_roof.n.01

puffbird.n.01

spiny_puffer.n.01

tortoise.n.01

hummingbird.n.01 clown_anemone_fish.n.01

salamander.n.02

restroom

carapace.n.01

lionfish.n.01

mackerel_shark.n.01

supervisoroffice casemeetingroom clientconsultationroom

aquarium.n.01

marble.n.03 marble.n.01

surgeonfish.n.01

goldfish.n.01

drawstring_bag.n.01

turk's_head.n.01

belt_buckle.n.01

clutch_bag.n.01

briefcase.n.01

publicservicecounter hotdeskzone registrationcheck-inarea breakoutdiscussionroom

accessory.n.01

accessory.n.02

shoulder_bag.n.01

greenhouse.n.01

satchel.n.01

windsor_eyeglasses.n.01

walking_stick.n.01

digital_watch.n.01

pocket_watch.n.01

key_ring.n.01

distillery

backpack.n.01

granite.n.01 paving_stone.n.01

key_fob.n.01

coffeeroastery

umbrella.n.01

wristwatch.n.01

###### sunglasses.n.01

spectacles.n.01

lattice.n.03

stone_wall.n.01 roughcast.n.02

lockerarea

smallworkshop

kettle.n.01 dishwasher.n.01 charcoal_burner.n.02

lab qualitylab qualitycontrol smallclassroom

can_opener.n.01 electric_refrigerator.n.01 water_cooler.n.01 propjet.n.01 deep-freeze.n.01 squeezer.n.01 reamer.n.01 mandolin.n.01 washer.n.02 pop-up_toaster.n.01 electric_oven.n.01 kitchen_appliance.n.01 toaster_oven.n.01 toaster.n.02 electric_mixer.n.01

churn.n.01 clothes_dryer.n.01

interviewroom

meat_grinder.n.01

window.n.01 timber.n.02

securerecordroom electronicslab legallibrary

decoratingroom

bender.n.01

interior_door.n.01 railing.n.01

blending coldconditioningroom labelingroom

simultaneoustranslationbooth trainingroom

cuppingroom

stove.n.01 stand_mixer.n.01 samovar.n.01 range_hood.n.01

finishingroom

electric_range.n.01

preproom mixingpreparationroom

tastingroom

doorknob.n.01 windmill.n.01

teakettle.n.01

coldstorageroom

coffee_mill.n.01

stove.n.02

gas_range.n.01

cooler.n.01

blender.n.01

refrigerator.n.01

coffee_maker.n.01

server.n.03

bubble.n.04

saltingroom doughpreparationroom ingredientpreproom spiritagingroom spiritmaturationroom

grinder.n.04

dispenser.n.01

pastryroom

water_heater.n.01

appliance.n.02

dome.n.01 chalet.n.01 column.n.07 geodesic_dome.n.01

water_filter.n.01

washer.n.03

indoorplaycenter

vending_machine.n.01

window_frame.n.01 stairway.n.01

securedfileroom modelworkshop

microwave.n.02

documentreviewroom smalleventarea materialsamplelibrary 3dprintingstudio controlroom itsupportroom smallinterviewroom phonequietpod feedbackroom phoneboothpod speakergreenroom

andesite.n.01 pagoda.n.01

appliance.n.01

rudaceous_rock.n.01 cristobalite.n.01 shingle.n.03

sandstone.n.01

bowlingalley indoorclimbinggym

pick.n.07

proofingroom

triangular_prism.n.01 ashlar.n.01

stanchion.n.01 sandwich_board.n.01 faceplate.n.01

maltmillingroom

stillroom greenbeanstorage cheesestorageroom

mannequin.n.02

quartz.n.02

briningroom

neolith.n.01

limestone.n.01

smallgym

bluestone.n.01

louver.n.01

standee.n.02

fixture.n.01

mounting.n.02

corrugated_iron.n.01 section.n.04 building_material.n.01

display.n.02

doorway.n.01

brownstone.n.01

cinder_block.n.01

tarpaulin.n.01

booth.n.02

milkpreparationroom

building_complex.n.01 sliding_door.n.01 plumbing.n.01 corbel.n.01 clock_tower.n.01 shale.n.01 volcanic_rock.n.01 cylinder.n.03 roman_arch.n.01 tunnel.n.01 farm_building.n.01 standpipe.n.01 housing.n.02 structural_member.n.01 cover_plate.n.01 raw_wood.n.01 hovel.n.01 telephone_pole.n.01 scaffolding.n.01 brickwork.n.01

display.n.06

partition.n.01

timber.n.03

baluster.n.01

chimney.n.01

skyscraper.n.01

rotunda.n.01

garage.n.01

curdprocessingroom

mock-up.n.01

community recreation center

acoustic_absorption.n.01 fortress.n.01 prefab.n.01

ripeningroom

designcritiqueroom

sedimentary_rock.n.01 stonework.n.01 i-beam.n.01 portcullis.n.01 manhole_cover.n.01 corner.n.09 chlorite.n.01 mausoleum.n.01 basalt.n.01 caisson.n.03 sliding_window.n.01 bridge.n.03

designpin-uproom relaxationroom coachingroom cloakroomstorage

pargeting.n.02

doorframe.n.01 gravel.n.01

ridge_tile.n.01

concrete_mixer.n.01 dry_wall.n.02 dowel.n.01 scaffold.n.02 amphibole.n.01 roman_building.n.01

portal.n.01

apartment_building.n.01 housetop.n.01 castle.n.03 battlement.n.01

agingroom

hammer.n.02

gothic_arch.n.01 cement.n.02

fermentationroom

suspension_bridge.n.01 cylinder.n.02 gantry.n.01 matchwood.n.01 t-junction.n.01 mineral_wool.n.01 water_tower.n.01

mechanical_device.n.01

pantile.n.01

storehouse.n.01 shelter.n.02 insulant.n.01

pointed_arch.n.01 flooring.n.02 stairs.n.01 tile_roof.n.01 mount.n.04 carport.n.01 bell_tower.n.01

excavation.n.03 breccia.n.01

concrete.n.01 base.n.19 baseboard.n.01

recreationsportsfacility

round_arch.n.01

bridge.n.01

wood.n.01

speakerpreproom printingroom avroom luggagestorage waitingroom finishingroom paintroom sewingroom interrogationroom podcastingbooth textileroom mother'sroom

abacus.n.02 regular_icosahedron.n.01 hemisphere.n.02 circle.n.08

plottingroom

japan.n.03 paperback_book.n.01 bible.n.01 structural_formula.n.01 oxford.n.04

naproom

packagingroom

kurdistan.n.02

temple.n.03 manuscript.n.02 booklet.n.01

foliation.n.03

orrery.n.01

coffee-table_book.n.01

numeral.n.01

hardback.n.01

torus.n.01

double_helix.n.01

tome.n.01

book.n.01

contour_map.n.01

elevation.n.07

celestial_globe.n.01

bottlingroom

monitoringroom dispatchroom

wellnessroom

hatchet.n.02

recordingbooth printroom

book.n.02

documentroom

globe.n.03

mailroom

archiveroom

copyroom

double-bitted_ax.n.01

galleryroom

administration

office

beer_bottle.n.01 drinking_fountain.n.01

fitnesscenter

carafe.n.01 beer_barrel.n.01

kylix.n.01 cheval_glass.n.01

wine_bottle.n.01

flagon.n.01 pop_bottle.n.01

demijohn.n.01

zarf.n.01 garibaldi.n.02

shaker.n.03

cup.n.08 coffee_mug.n.01 coaster.n.03 hand_glass.n.01 water_bottle.n.01 hookah.n.01

roastingroom

wineglass.n.01

dixie_cup.n.01

tankard.n.01 drinking_vessel.n.01 beer_mug.n.01

teacup.n.02

hand_ax.n.01 tool.n.01 mechanism.n.05 die.n.01 bracket.n.04

beer_can.n.01

single_prop.n.01

musical instrument store

coffee_cup.n.01

unloadingarea

chalice.n.01

goblet.n.01

cannikin.n.01

soda_can.n.01

cup.n.01

mug.n.04

loadingarea

dishwashingarea

bugle.n.01 stringed_instrument.n.01 harp.n.01 fipple_flute.n.01 sputnik.n.01

iron_maiden.n.01 piano_keyboard.n.01

woodworking shop

keyboard_instrument.n.01

bell.n.05 balalaika.n.01 music_stand.n.01 whistle.n.03 flute.n.01

bookstore with cafe

band.n.11 accordion.n.01 recorder.n.01

marimba.n.01

cello.n.01 harmonica.n.01 metronome.n.01 crotchet.n.04

panpipe.n.01

gong.n.01

grand_piano.n.01

music_box.n.01

samisen.n.01

cornu.n.01

ocarina.n.01

lyre.n.01 handbell.n.01 brass.n.05

instrument.n.02

bass_guitar.n.01

component.n.03 scythe.n.01 machete.n.01 carpenter's_hammer.n.01

banjo.n.01 synthesizer.n.02

maraca.n.01

drum.n.01

lute.n.02

horn.n.06

acoustic_guitar.n.01

potterystudio

cornet.n.01

guitar.n.01

violin.n.01

horn.n.08

drum.n.04

electric_guitar.n.01

brass.n.07

freezerroom

exhibitroom

pocketknife.n.01 machine.n.01

pipe_wrench.n.01 hand_tool.n.01

shelter.n.01 protective_covering.n.01

spacesuit.n.01 movable_barrier.n.01 protective_garment.n.01

birdcage.n.01 chainlink_fence.n.01

cage.n.01 diving_suit.n.01 enclosure.n.01 ocular_helmet.n.01 barricade.n.02

cleaningarea

thriftstore

gun_enclosure.n.01

rod.n.01 valve.n.03

fence.n.01

shield.n.01

space_helmet.n.01

flat_tip_screwdriver.n.01 angle_bracket.n.02 handle.n.01 screw.n.04 electric_drill.n.01 machine_bolt.n.01 hook.n.05 shard.n.01 crowbar.n.01 hammer_axe.n.01 pliers.n.01 carving.n.01 phillips_screw.n.01 machine_tool.n.01 screw_wrench.n.01 stick.n.02

barrier.n.01

refrigerationroom

mallet.n.03 molding.n.02

tailor shop sewing workshop repair shop retail store jewelry workshop indoor market arcade toy store

shield.n.02

department store

charm.n.03 pendant_earring.n.01 emerald.n.01 chain.n.10 opaque_gem.n.01 beads.n.01 gold.n.01 tooth_shell.n.01 ruby.n.02 tortoiseshell.n.01 signet_ring.n.01 choker.n.03

storagepantry

chrysoprase.n.01

bloodstone.n.01

emerald.n.02

medallion.n.02

ruby.n.01 medallion.n.05 bead.n.01 citrine.n.01 goldstone.n.01

beading.n.01

sapphire.n.01

brooch.n.01 cupid's_bow.n.02

amulet.n.01

circlet.n.02 medallion.n.03

chain_saw.n.01 wellhead.n.02

tiara.n.01

filigree.n.01

lunula.n.02

wheelwork.n.01 sickle.n.01 jaw.n.01 ice_ax.n.01 power_tool.n.01 open-end_wrench.n.01

ring.n.02 turquoise.n.01 medallion.n.01

gem.n.02

chain.n.03

necklace.n.01

fireman's_ax.n.01

electric_motor.n.01

amethyst.n.01

grocery store

anchor.n.01

mainproductionkitchen

escutcheon.n.03

crown.n.04

restroom entrance lobby

toolbox.n.01 clamp.n.01

ax_head.n.01 flint.n.01 handsaw.n.01

screwdriver.n.01 blowtorch.n.01 standard.n.05

pendant.n.01

power_drill.n.01

storageroom

scepter.n.02

ball-peen_hammer.n.01 wrench.n.03 rotating_mechanism.n.01 mortar.n.03 bobbin.n.01 compass.n.01

ball_valve.n.01

bracelet.n.02

broadax.n.01 ladder.n.01

carpenter's_mallet.n.01 outlet_box.n.01 hinge.n.01 pipefitting.n.01 spike.n.09 grindstone.n.01

toroid.n.01

scissors.n.01

stick.n.07

air_compressor.n.01 hand_pump.n.01 millstone.n.03

ring.n.08

dipper.n.01 cutting_implement.n.01 wooden_spoon.n.02 ladle.n.01 peeler.n.03 slice.n.05 opener.n.03 fork.n.01 kitchen_utensil.n.01 spatula.n.02 whisk.n.01

coil_spring.n.01

holding_device.n.01 knob.n.02

restroom

scoop.n.05 butter_knife.n.01

needlenose_pliers.n.01

shaping_tool.n.01 hand_drill.n.01

spoon.n.01 knife_blade.n.01

grate.n.03 bottle_opener.n.01 corkscrew.n.01 chopping_knife.n.01

nutcracker.n.01

stick.n.01 cookie_cutter.n.01 scraper.n.01

chisel.n.01

drill.n.01

abrading_stone.n.01 turbine.n.01 measuring_stick.n.01 staple_gun.n.01

pit.n.01 skewer.n.01

saw.n.02

tongs.n.01 butcher_knife.n.01 sallet.n.01

spatula.n.01

spray_paint.n.01 bellows.n.01

actuator.n.01

adhesive_tape.n.01 pulley.n.01 circuit_breaker.n.01 hex_nut.n.01 duct.n.03 nozzle.n.01 hammerhead.n.03 portable_circular_saw.n.01

nailhead.n.01

iron.n.04

pestle.n.02

theodolite.n.01

shovel.n.01

whip.n.01

pestle.n.03

fork.n.04

nail.n.02

phillips_screwdriver.n.01 pipe_cutter.n.01 piston.n.02 coupling.n.02 vise.n.01 c-clamp.n.01 bolt_cutter.n.01 wood_chisel.n.01 gearset.n.01 butt_hinge.n.01 magnet.n.01 conveyer_belt.n.01 sextant.n.02 steel_trap.n.02 grease_gun.n.01 d4.n.01 sawhorse.n.01 screw.n.02 spray_gun.n.01

paint_roller.n.01 steel.n.03 skeleton_key.n.01 pump.n.01

spiral.n.04 rolling_pin.n.01

equipment.n.01

- knife.n.01
- knife.n.02

awl.n.01

case_knife.n.02

wedge.n.01 coil.n.01

cleaver.n.01

soldering_iron.n.01 hoe.n.01 center_punch.n.01 caliper.n.01

oilcan.n.01

connecting_rod.n.01 level.n.05 locking_pliers.n.01 twist_bit.n.01 claw.n.01 butterfly_valve.n.01 glass_cutter.n.03 compass.n.04 rope.n.01 hacksaw.n.01 saber_saw.n.01 lever.n.01 micrometer.n.02 hook.n.04 crosshead.n.02 spindle.n.02 sewing_machine.n.01 power_shovel.n.01 plow.n.01

hoist.n.01

peg.n.04

wire_cutter.n.01

magnetic_compass.n.01

wire_stripper.n.01 hook.n.02 whetstone.n.01 crescent_wrench.n.01 stillson_wrench.n.01 hammer_and_sickle.n.01 pitchfork.n.01 lever.n.02

adjustable_wrench.n.01 socket_wrench.n.01

clockwork.n.01

winery

centrifugal_pump.n.01 bench_clamp.n.01

blade.n.09

air_hammer.n.01

metal_screw.n.01

meterstick.n.01

cylinder_head.n.01 mortar.n.01 anvil.n.01 swatter.n.01 gauge.n.01

pegboard.n.01 dig.n.01 handwheel.n.02 slip-joint_pliers.n.01

allen_screw.n.01

sheet_metal.n.01

woodscrew.n.01

step_ladder.n.01

linkage.n.03

piling_hammer.n.01

grinding_wheel.n.01 fishplate.n.01

power_saw.n.01

winch.n.01

circular_saw.n.01

pressure_gauge.n.01

lifting_device.n.01

vernier_caliper.n.01 impeller.n.01

scale.n.07

extension_cord.n.01

mousetrap.n.01

machine_screw.n.01

drone.n.04 sports_car.n.01

box_wrench.n.01 blade.n.08

wedge.n.06

wing_nut.n.02

worm_gear.n.01 trap.n.01

allen_wrench.n.01

ball_bearing.n.01 winder.n.02

drilling_bit.n.01

cylinder.n.01

sprocket.n.02

miter.n.03

hardware.n.01

air_pump.n.01

toggle_bolt.n.01

extrusion.n.02

pallet.n.02

fishhook.n.01

bandsaw.n.01 knob.n.01

grapnel.n.01

reel.n.03

swivel.n.01

dolly.n.02

press.n.03

cafe

stall.n.03 sugar_bowl.n.01

soup_plate.n.01

dish.n.01 serving_dish.n.01 platter.n.01 tablefork.n.01 soup_bowl.n.01 bowl.n.02 plate.n.02

bowl.n.03

plate.n.04

table.n.02

starship.n.01

bowl.n.01

ticketreception

dish.n.05 mixing_faucet.n.01 basket.n.03 round_bone.n.01 mold.n.02 funnel.n.02 saucepan.n.01 countertop.n.01 wishbone.n.01 stoup.n.02 pepper_mill.n.01 wine_cask.n.01 milk_can.n.01 pan.n.01 dish_rack.n.01 cork.n.04 icetray.n.01 willowware.n.01 plat.n.01 trivet.n.02

assessmentroom promerchandiseroom physiotherapyroom

funnel.n.01 water_jug.n.01

kitchen_island.n.01

ax.n.01

quietroom consultationroom wellnessroom testingroom

gamesroom

meeting musicstudio

lid.n.02 ceramic_ware.n.01

frying_pan.n.01

oyster_shell.n.01

loungeroom

sensoryroom

dancestudio

shell.n.08

caldron.n.01 kitchen_sink.n.01 earthenware.n.01

classroom

saunaroom

introroom trainingboardroom cyclingroom childcareroom refreshmentarea spinningroom

urn.n.01

steamroom

proshoproom

bell.n.01

massageroom

nutritionconsultationroom

yogaroom stretchingroom briefingroom campusboardroom

airplane.n.01 warplane.n.01

car.n.01

sink.n.01

pot.n.04

pitcher.n.02

gameroom viplanelounge arts&craftsroom climbingshoestorage

train_set.n.01

yoganook stretchingnook

tray.n.01

ticketdesk

vessel.n.03

educationspace

vendingarea

crock.n.03

animatronics.n.01

jug.n.01

arcade.n.02 slot_machine.n.01 amusement_arcade.n.01 ferris_wheel.n.01

cone.n.01

biplane.n.01

werewolf.n.01 pinball_machine.n.01

car_wheel.n.01 tank.n.01

clapperboard.n.01

black_widow.n.01

groupexercisestudio

snackarea parents'lounge boulderingcave

arcaderoom

maze.n.01 firework.n.01

fantasy.n.02

propeller_plane.n.01

arcade_video_game.n.01

lightstick.n.01

mermaid.n.01

zombi.n.03

helicopter.n.01

sights.n.01

ghost.n.03

jukebox.n.01

personaltrainingroom

planetarium.n.03

toddlers'playroom

cameo.n.01

scorekeepers'booth

fictional_character.n.01

cafearea

grab.n.01

classroomspace giftbookshop staffroom

sedan.n.01 vehicle.n.01 airliner.n.01 sport_utility.n.01

monster.n.01

privatepartyroom

birthdaypartyroom

giftshop

spacecraft.n.01

rowing_boat.n.01 tire.n.01 bicycle.n.01 military_vehicle.n.01 armored_vehicle.n.01

administration

aircraft.n.01

handcart.n.01 car_tire.n.01

delivery_truck.n.01

club.n.06 bicycle_pump.n.01 gymnastic_apparatus.n.01

jet.n.01

office

stopwatch.n.01 table-tennis_racquet.n.01 cricket_bat.n.01 paddle.n.01 boxing_glove.n.01 puck.n.02

multiengine_airplane.n.01 wagon_wheel.n.01 hot_rod.n.01 beach_wagon.n.01 wagon.n.04

gas_pump.n.01 drive.n.10 wheel.n.01 armored_car.n.02 twinjet.n.01 jeep.n.01 hovercraft.n.01 plane.n.05 pickup.n.01 engine.n.01 monoplane.n.01 steering_wheel.n.01 hatchback.n.01

truck.n.01

van.n.05

administrativeroom collectionroom

baton.n.04 scoreboard.n.01 baseball_glove.n.01 hang_glider.n.02

triton.n.03 rim.n.03

bat.n.01 cartwheel.n.01 mountain_bike.n.01

jump_rope.n.01 in-line_skate.n.01

exercise_device.n.01

staffroom

oar.n.01 whiteface.n.02

hockey_stick.n.01

barbell.n.01 tennis_racket.n.01 fishing_rod.n.01

club.n.03 runner.n.08

space_shuttle.n.01 barn.n.01 chassis.n.03 motorcycle.n.01 cruiser.n.01

roller.n.04 boomerang.n.01

hoop.n.02 football_helmet.n.01 golf_ball.n.01 bowling_ball.n.01 football.n.02 surfboard.n.01 tennis_ball.n.01 baseball.n.02

trailer_truck.n.01

volleyball.n.02

wheeled_vehicle.n.01 panda_car.n.01

###### jumping_jack.n.01

bicycle_wheel.n.01

covered_wagon.n.01 gearbox.n.01 blimp.n.02 stealth_aircraft.n.01 ambulance.n.01 self-propelled_vehicle.n.01

shock_absorber.n.01 tractor.n.01 spark_plug.n.01

bowling_pin.n.01

bat.n.05

floatplane.n.01 castle.n.02 delta_wing.n.01 tracked_vehicle.n.01

canoe.n.01

punching_bag.n.02

shuttlecock.n.01

automobile_engine.n.01 motor_scooter.n.01

outboard_motorboat.n.01 hoverboard.n.01 propeller.n.01 dashboard.n.02 trailer.n.03 office_building.n.01

passenger_van.n.01

###### barrel.n.02 crate.n.01 box.n.01

skateboard.n.01

submarine.n.01 tricycle.n.01

reconnaissance_vehicle.n.01

crane.n.04

wheel.n.04

float.n.06

basketball.n.02

convertible.n.01 used-car.n.01 steamroller.n.02

space_capsule.n.01 minicar.n.01 brake_disk.n.01 market_cross.n.01

stock_car.n.02 lander.n.02 snow_tire.n.01

bus.n.01

outside_mirror.n.01 trailer.n.04 railroad_track.n.01 watchtower.n.01 airplane_propeller.n.01 six-spot.n.01

repulsorlift_engine.n.01 jet_propulsion.n.01 camshaft.n.01 traction_engine.n.01 trail_bike.n.01 supercharger.n.01

hot-air_balloon.n.01 caterpillar.n.01 hardtop.n.01 steam_engine.n.01

bow.n.06 racer.n.02 dumbbell.n.01

dump_truck.n.01

exhaust_manifold.n.01

airship.n.01

engine_block.n.01 bumper.n.02 small_boat.n.01

jet_engine.n.01

tractor.n.02

tanker_truck.n.01 auto_part.n.01

passenger_car.n.01 hull.n.06 handlebar.n.01

check-in

baseball_bat.n.01

suspension.n.05

fly.n.01

license_plate.n.01 boxcar.n.01 dune_buggy.n.01

transmission.n.05 brake.n.01 go-kart.n.01 dock.n.04 cockpit.n.01

bulldozer.n.01

ladder_truck.n.01 streetcar.n.01 glider.n.01

van.n.04

speedboat.n.01 scooter.n.02

seaplane.n.01

applecart.n.02 horse.n.01 minivan.n.01

rail.n.04

carriage.n.02

bodywork.n.01 fender.n.01

soccer_ball.n.01

tramcar.n.01

activityhall

elevator.n.01

chassis.n.02

minisub.n.01 hub.n.01

wagon.n.01

boat.n.01

ball.n.03

storageroom

hammock.n.02 mountain_tent.n.01

hands-onexperimentlab demonstrationroom

ladybug.n.01 weathervane.n.01 field_tent.n.01 parasol.n.01 acorn.n.01

bird_feeder.n.01

turf.n.01 clothesline.n.01

tent.n.01 sleeping_bag.n.01 firewood.n.01

tree_house.n.01

wood_ant.n.01

kayak.n.01 barbecue.n.03

hay_bale.n.01

camper.n.02

carabiner.n.01

bicycle_rack.n.01

changingroom equipmentstorageroom

mainhall

sled.n.01 pavilion.n.01

bait.n.02 birdbath.n.01 tree.n.02 campfire.n.01 mound.n.04

postbox.n.01

dumpster.n.01 swimming_pool.n.01 scarecrow.n.01 canopy.n.03 boulder.n.01 flagpole.n.02 starfish.n.01 ramp.n.01 fire_pit.n.01

terrain.n.01 birdhouse.n.01

lilo.n.01

bollard.n.01

butterfly_fish.n.01

starprojectorbooth arts&craftsstudio

gazebo.n.01 coral.n.04 picket_fence.n.01

stump.n.01

vrspacesimulationroom telescopecontrolroom

###### pebble.n.01

reception

tree.n.01

smalltechnicalsupportroom pressinterviewroom specimenidentificationroom

sensoryplayroom

explorationtunnel

treelet.n.01

astronomyworkshop artconsultingoffice puppettheaterroom

storytellingnook

gymhall

parachute.n.01 combination_lock.n.01

release.n.08 signaling_device.n.01

security_system.n.02

beacon.n.03 bulletproof_vest.n.01 panic_button.n.01

parking_meter.n.01

temporaryorganizeroffice privatesalesroom photographysetuproom

padlock.n.01 crash_barrier.n.01 biohazard_suit.n.01

fire_screen.n.01

safety_rail.n.01

life_jacket.n.01

exhibitorservicebooth fossilpreparationlab

safety_pin.n.01

knee_pad.n.01

doorlock.n.01

taxidermyworkshop microscopysuite

guard.n.03 life_buoy.n.01

roadblock.n.02

warning.n.01

photodocumentationroom artist-in-residencestudio

earplug.n.01

goggles.n.01 gasmask.n.01 crash_helmet.n.01

lockup.n.01

fire_alarm.n.02

fire_extinguisher.n.01

hard_hat.n.02

###### container.n.01

fireplug.n.01

body_armor.n.01

false_face.n.01

meetingroom

###### chandelier.n.01

observationpreproom roboticsworkshop

mask.n.01

breakoutroom

helmet.n.01

portfolioreviewlounge privateviewingroom

helmet.n.02

framingworkshop

vrsimulationbooth makerspace

treasure_chest.n.01 amphora.n.01 rack.n.04 wicker_basket.n.01

arsimulationbooth artistpreproom

bottle.n.01 ashcan.n.01

curationworkspace laboratoryroom conservationroom

double-breasted_jacket.n.01

cargo_container.n.01

hackerspace

lockerroom

main service area

distressed_jeans.n.01

jewelled_headdress.n.01

restroom

queen.n.08 dinner_dress.n.01 strip.n.05 hair_slide.n.01 hand.n.08

photograph_album.n.01

loungewear.n.01

strip.n.02 bolero_hat.n.01 gauntlet.n.03

jersey.n.04 business_suit.n.01

cart.n.01 bin.n.01

shopping_cart.n.01 pouch.n.01 clothes_hamper.n.01

turtleneck.n.01

greatcoat.n.01

stud.n.02 long_trousers.n.01 cardigan.n.01 pickelhaube.n.01

stretch_pants.n.01

recycling_bin.n.01 cereal_box.n.01 carryall.n.01

sweat_suit.n.01

ski_mask.n.01

kepi.n.01 sailor_cap.n.01

jump_suit.n.02

cap.n.02 dunce_cap.n.01 hood.n.09

balaclava.n.01

storage_space.n.01 toy_box.n.01 receptacle.n.01

patch.n.03 sundress.n.01

shopping_bag.n.01 shoebox.n.01 gunnysack.n.01 herm.n.01

button_ad.n.01

skirt.n.02 wet_suit.n.01

head.n.01 brassiere.n.01 coat_hanger.n.01

holder.n.01 case.n.20 case.n.05

cap.n.04 stirrup.n.01

capote.n.02

harness.n.01

lanyard.n.02

trench.n.03

headdress.n.01

fur_hat.n.01

bomber.n.03

button.n.01

riband.n.01

canister.n.02 barrel.n.03 safe.n.01

slacks.n.01

sleeve.n.02

collar.n.03

watch_cap.n.01 jump_suit.n.01 trench_coat.n.01

necktie.n.01

skullcap.n.01

jar.n.01

cape.n.02

restroom

jumper.n.05 bandanna.n.01

scarf.n.01

blazer.n.01

cloak.n.02 running_suit.n.01

jaw.n.03

sock.n.01

sweat_pants.n.01

strongbox.n.01

button.n.08

collar.n.06

camouflage.n.02

bread-bin.n.01

table_lamp.n.01 lantern.n.01 lighting_fixture.n.01 floor_lamp.n.01

hide.n.01 maillot.n.01 arm.n.01 hanger.n.02

garment.n.01

pith_hat.n.01

cross.n.01

limb.n.02 kimono.n.01

blouse.n.01 picture_hat.n.01

bowler_hat.n.01

cocktail_dress.n.01

shoebox.n.02

pin.n.01

leg.n.03

tricorn.n.01

bomber.n.01 tank_top.n.01 short_pants.n.01 clothes_tree.n.01

arm.n.02 bow_tie.n.01

vest.n.01 trousers.n.01 gown.n.05

sweater.n.01 sportswear.n.01

coffer.n.01

coat.n.01 wishing_cap.n.01

stob.n.01

sack.n.01

strap.n.01

elbow.n.03

pile.n.01

gown.n.01

parka.n.01

robe.n.01

gauntlet.n.02

cap.n.01 sweatshirt.n.01 polo_shirt.n.01 fedora.n.01 costume.n.01 badge.n.01 bomber_jacket.n.01 cowboy_hat.n.01 belt.n.02 dress_hat.n.01

suit.n.01

beanie.n.01

hand.n.01 button.n.04 sombrero.n.02 headband.n.01

outfit.n.02

jeans.n.01

wig.n.01 button.n.07 sunhat.n.01

cross.n.03 fursuit.n.01 cigar_lighter.n.01

clothing.n.01

glove.n.02

dress.n.01 headpiece.n.02

hat.n.01 baseball_cap.n.01

jacket.n.01

common_mackerel.n.01

patty.n.03 ribbonfish.n.02 cheshire_cheese.n.01

jersey.n.03

buttercup_squash.n.02

shirt.n.01

pepperoni_pizza.n.01

pike.n.04 canned_food.n.01 corn.n.03 octopus.n.02

sweet_orange.n.01 japanese_banana.n.01 french_bread.n.01

golden_delicious.n.01

savoy_cabbage.n.01

peanut.n.04 canned_meat.n.01 baguet.n.01

dark_bread.n.01 cabbage.n.03 cauliflower.n.01

sausage.n.01 fish_species.n.01

carrot.n.01 watermelon.n.02 butternut_squash.n.02

mushroom.n.01

grapefruit.n.02 purple_onion.n.01

acorn_squash.n.02

honeycomb.n.02

monkfish.n.01

beverage.n.01

community event room

squid.n.02 corncob.n.01

sea_urchin.n.01

flatbread.n.01

fruit.n.01 cucumber.n.02 sweet_potato.n.02

chili.n.02 mealie.n.01

lobster.n.02

hamburger.n.01

shell.n.01 nugget.n.01 stockfish.n.01

quince.n.02

coconut.n.02

bosc.n.01 banana.n.01

pizza.n.01 sturgeon.n.01

pita.n.01 truffle.n.03

cheese.n.01

egg.n.02 goosefish.n.01 bagel.n.01

g.n.09 stocks.n.03

e.n.05 steak.n.01

broccoli.n.01

bartlett.n.03

bread.n.01

chicken.n.02

taco.n.02

bleu.n.01

finishingroom electronics testing room ironingroom birthday party room pressingroom item cleaning room

brie.n.01

granny_smith.n.01

litchi.n.02

anjou.n.02

swiss_cheese.n.01

zucchini.n.02

lime.n.06

pod.n.02 gourd.n.02

hog.n.03

hotdog.n.02

sushi.n.01 clamshell.n.01

garlic.n.01

pomegranate.n.02

fig.n.04

strawberry.n.01

grape.n.01

cantaloup.n.02

privateroom token counting room staffworkshop demoroom

raw_meat.n.01

walnut.n.01

a.n.06

food.n.02

sandwich.n.01

gourd.n.01

crepe.n.01

flashlight.n.01

duck.n.01

onion.n.01

kiwi.n.03

ginger.n.03

mat.n.03

ice-cream_cone.n.01

garlic.n.02

recording demo booth

mango.n.02 peach.n.03 pineapple.n.02

light_arm.n.01

common_ax.n.01 potato.n.01 bell_pepper.n.02

egg.n.01

bar.n.03 pear.n.01 tomato.n.01

toy assembly room

crab.n.01 broccoli.n.02

pod.n.01

ticket counting room

lemon.n.01 stake.n.05 cheeseburger.n.01

mat.n.01 french_loaf.n.01 core.n.05

avocado.n.01

bakerykitchen tool sharpening room volunteeroffice author signing room

sample.n.03

roll.n.11

signboard.n.01

banana.n.02 loaf_of_bread.n.01 mushroom.n.03 mushroom.n.02

###### cabinet.n.01 chest.n.02 figure.n.04

pattern drafting area

orange.n.01

small event room

fittingstudio small shared kitchen floral prep room cold storage room deli preparation room

children's playroom

taggingarea

photostudio

skull_and_crossbones.n.01

viplounge wood storage room

pricingarea

market manager office vendor preparation room

practiceroom sheet music library pattern making room customer service kiosk

playroom photobooth

student_lamp.n.01 oil_lamp.n.01 candle.n.01 streetlight.n.01 lamppost.n.01 light_bulb.n.01

butcherroom parcel pickup room

apple.n.01

pumpkin.n.02

sign.n.02

sconce.n.04 lamp.n.01

gummed_label.n.01

hurricane_lamp.n.01

chest_of_drawers.n.01

fluorescent_lamp.n.01 fairy_light.n.01

direction_sign.n.01

kiln firing room customer waiting lounge material cutting room glaze mixing room clay mixing room game repair workshop readingroom donation sorting room readingnook tool cleaning room

candlestick.n.01 lamp.n.02

fish.n.01

magic_lantern.n.01

lampshade.n.01 beam.n.02

candelabrum.n.01 sconce.n.03

signpost.n.01 traffic_light.n.01 stop_sign.n.01

partyroom

light_fitting.n.01

billboard.n.01

small vault room arcade machine storage piece drying room equipment testing room

fluorescent.n.01

emblem.n.02

insignia.n.01

searchlight.n.01

###### houseplant.n.01 driftwood.n.01

street_sign.n.01

banner.n.01 flag.n.06 signage.n.01

symbol.n.02

spotlight.n.02

embroidery room

flambeau.n.01 bulb.n.04

skylight.n.01

nameplate.n.01

lighting.n.02

flag.n.04

torch.n.01

flag.n.01

unidentified_flying_object.n.01

logo.n.01

light.n.02

flash.n.09

bulb.n.03

customer pickup room design consultation room

etagere.n.01

hypothetical_creature.n.01

regular_dodecahedron.n.01

extraterrestrial_being.n.01

geological_formation.n.01

quadrangular_prism.n.01

time_machine.n.01 long-horned_beetle.n.01

cabinet.n.04

roman_candle.n.01 carnivorous_plant.n.01

###### christmas_tree.n.05

refrigerated storage

horseshoe_crab.n.01

fabric cutting room children's story area finished goods storage

elephant.n.01 apparition.n.02 arabic_numeral.n.01

japanese_flowering_cherry.n.01

###### object.n.01

grandfather_clock.n.01

agglomeration.n.01

natural_object.n.01

animal.n.01 creature.n.02 telephone_booth.n.01

parts storage room

fabric storage room

microcosm.n.01

scrap_metal.n.01

assembly.n.01 dodecahedron.n.01

baby_buggy.n.01

bedroom_furniture.n.01 banquette.n.01

project assembly area consultation room prize redemption room

nodule.n.03 firecracker.n.01

buffing room lessonroom small safe room

instrument repair room measurement room soundproof testing room

polyhedron.n.01

staffroom

reception area waiting area service room treatment room utility room storage room wine aging cellar wine display area service_treatment_room main_service_area reception_waiting_area

stained-glass_window.n.01

arthropod.n.01

giraffe.n.01 deadwood.n.01

badass.n.02 cockroach.n.01

polishing room

moth.n.01 centipede.n.01

thing.n.04 impression.n.07

mythical_being.n.01

overall.n.01 stegosaur.n.01

smiley.n.01 apparition.n.01

velociraptor.n.01

flower_arrangement.n.01

pedestal.n.03 basket.n.01 cushion.n.03 platform.n.01

magazine_rack.n.01

spinning_wheel.n.01

corner_cabinet.n.01 dresser.n.05 crib.n.01

coffee_table.n.01 ottoman.n.03 coatrack.n.01 wardrobe.n.01 bungalow.n.01 base.n.08 rail_fence.n.01 cabinet.n.03 casket.n.02 double_door.n.01

tapestry.n.03 cuckoo_clock.n.01 doorstop.n.01 ceiling.n.01 cathedral.n.01 shrine.n.01 kakemono.n.01 mantel.n.01 sunburst.n.03 shag_rug.n.01 fireplace.n.01 frieze.n.01 topiary.n.02 pilaster.n.01 analog_clock.n.01 bamboo.n.02 common_daisy.n.01 eye-catcher.n.01 delft.n.01 philodendron.n.01 interior_decoration.n.01 electric_iron.n.01 tulip.n.01

powder.n.01

stand.n.04 arch.n.03 log.n.01

reception_desk.n.01

monarch.n.02

china_cabinet.n.01

dressing_table.n.01 bannister.n.02

pendulum_clock.n.01 tudor_architecture.n.01

whirling_table.n.01 boulle.n.01

shelf_bracket.n.01 armchair.n.01

detail.n.02

tangle.n.01

serving_cart.n.01

debris.n.01

seat_cushion.n.01 board.n.03

platform_bed.n.01 support.n.07 highchair.n.01 headrest.n.02 bed.n.01 pad.n.04

fragment.n.01 abstraction.n.04

emoji.n.01 bug.n.01 lizard.n.01 cyborg.n.01

ovoid.n.01

property.n.05 chair.n.01 house.n.12

ranch_house.n.01 stabile.n.01 bookshelf.n.01 park_bench.n.01

column.n.04 board.n.02 desk.n.01

blob.n.01

headboard.n.01 coffin.n.01 sofa.n.01

part.n.01

matchwood.n.03 basinet.n.01

bird.n.01 person.n.02 butterfly.n.01

french_door.n.01

item.n.03

bookcase.n.01 bench.n.01

snail.n.01

item dropoff room

daffodil.n.01 persian_violet.n.01 weathercock.n.01 curtain_ring.n.01 cornice.n.03 amaryllis.n.01 rosebud.n.01 venetian_blind.n.01 pediment.n.01 frame.n.01 steam_iron.n.01 terrarium.n.01 stony_coral.n.01 hortensia.n.01 erose_leaf.n.01 fire_iron.n.01 fountain.n.04 dandelion.n.01 rose_window.n.01 jalousie.n.02 scatter_rug.n.01 grotto.n.01 fishbowl.n.02

chiffonier.n.01 trunk.n.02

top.n.08 person.n.01

easy_chair.n.01

icon.n.01

jack-o'-lantern.n.02 mirror.n.01 ceiling_dome.n.01 fountain.n.01 decoration.n.02 cornice.n.02 pinecone.n.01 clock.n.01 sprig.n.02

log_cabin.n.01

insect.n.01

bee.n.01

shelf.n.01

pillar_box.n.01

baldachin.n.01 villa.n.02

wall_unit.n.01

beanbag.n.01

sectional.n.01 lectern.n.01 stack.n.01 coupe.n.01 counter.n.03

furniture.n.01

pot.n.01

gnome.n.01 wainscot.n.02 bouquet.n.01 monstera.n.01 boston_fern.n.01

gondola.n.02

map.n.01

sansevieria.n.01

armrest.n.01

cabin.n.02

drawer.n.01

rug.n.01 pyramid.n.01 flower.n.01 brain_coral.n.01

wall_clock.n.01

stool.n.01

caster.n.03

house.n.01

well.n.01 mailbox.n.01 curtain.n.01 wall.n.07 poster.n.01 sandglass.n.01 cactus.n.01 finial.n.01 facade.n.01

pot_plant.n.01

trunk.n.01

coffer.n.02

room.n.01

sunburst.n.02

seat.n.03

knocker.n.05

changing room

bonsai.n.01

cattail.n.01

student work storage

sewing machine room

gem setting room

staff break room

casting room

cafe kitchen

fitting room

checkout area

stock room

register area

back office

storage for specialty drinks

counter area

order packing station food assembly area display case area beard grooming area

self_service_computer_area binding_finishing_room private meeting room private styling booth wine education room food pairing prep space barista training area play area for children private lounge room private vip room children's play room private_fax_scan_booth employee break room snack preparation room cocktail mixing lab dessert preparation area private dining room consultation room facial room

storage room

design_consultation_room

sales floor

drive-thru pickup room nail treatment room retail sales room

restroom

private tasting suite

storage_utility_room

hair wash station

photo_printing_booth makeup studio sommelier's wine room chef's table room

children's haircut room

game room (e.g. darts packaging area

game room (e.g. pool)

retail bottle shop

hair wash area

small testing room plating room

bottling room

staff_room

expediter room

small lab room

shaving room

labeling area

baking room

pastry room

∼110k MolmoSpaces-Scenes-MultiType scene specifications ∼129k Objaverse objects

- Figure 17 Scene specification distribution (left) –with generic and concrete scene types and room types– used to generate MolmoSpaces-Scenes-MultiType scenes. Between one and ten rooms of mostly scene-specific room types (here shown aggregated per generic scene type) are chosen to prompt LLMs. Right, distribution of WordNet synsets grouped by higher-level object categories in our curated subset of Objaverse.

We provide models from both AI2-THOR and Objaverse. We extracted and converted objects from AI2-THOR and migrated them into a file format compatible with MuJoCo as well as other simulators such as ManiSkill and IsaacSim. In total, we converted 1.6k rigid object instances across 134 categories into MuJoCo. In addition, there are 22 categories—including doors, refrigerators, and dressers—that were made articulable by annotating joint information, including joint type (slide or hinge), joint axis, joint position, and joint range.

To ensure physical realism for robot manipulation tasks, we performed several validation steps. For rigid objects, we verified that mass and density values were realistic by comparing simulated values with estimates annotated via large language models (LLMs), adjusting density as needed. For articulable objects, we created a teleoperation suite to manually tune the physical properties of joints and the density of movable parts. This was performed using a simulated Franka FR3 robot, which itself was tuned through system identification: real robot trajectories of picking and pushing cubes of known weights were collected, and simulation parameters were optimized to match the observed motions. These physics parameters were then applied to all objects during scene generation.

Collider meshes for all objects were generated using COACD [43] , and we also annotated primitive colliders for all THOR objects. For physics stability, rigid objects with receptacles (e.g., tables, dressers) primarily use primitive colliders to avoid mesh-mesh contact issues. Manipulable objects require higher fidelity, so convex decomposition was used; however, for very small and thin objects, primitive colliders were employed to maintain simulation stability. Meshes were further processed and decimated to improve simulation performance.

In addition to AI2-THOR objects, we converted a curated selection of Objaverse objects into the MJCF format for MuJoCo. This required a careful curation methodology to ensure quality and compatibility, which follows multiple filtering stages applied on top of an initial subset of 625k Objaverse objects pre-filtered with their original metadata, which are (1) converted to a format compatible with AI2-THOR [18, 44] and (2) annotated with VLM-generated [48] descriptions, estimated mass, canonical poses, pickable/receptacle properties, matching Wordnet synsets [45, 46], and scale estimates obtained via prompting with renderings from different viewpoints.

The automatic process results in some malformed or unreliable annotations, which prompts us to perform a complementary GPT-4.1 annotation (LLM prompt listed in Fig. 20) to determine (1) the type and number of object instances in

each model file, (2) the presence of props or other excessive geometry not part of the main object in the model file, (3) missing geometry, which typically occurs with scans of real-world objects, (4) a texture quality score on a 0-9 scale, and (5) whether the model contains receptacle parts. While we seek extraction of very specific information to help filter objects towards usability for embodied agent training, we note other efforts in re-annotating Objaverse like [61].

To provide a subset of objects suitable for LLM-backed scene generation, we sequentially apply filtering stages to ensure

- (1) metadata completeness and synset validity, (2) presence of a single object in the model file, (3) statistical scale inliers using a restrictive Tukey rule on the IQR of the log-scale, (4) sufficient texture quality with annotated score 4 or higher, (5) cross–renderer fidelity with CLIP [62–64] similarity score 0.6 or higher, (6) processed model file size with compressed geometry and colliders less than 1.5 MB, (7) agreement on the receptacle property in both annotations, (8) watertight colliders with at least 80% of the horizontal surface for receptacle objects, and (9) final removal of singleton representatives of synsets. The resulting distribution of curated objects, corresponding to almost 2.8k different WordNet synsets, is illustrated by Fig. 17 (right). The count of nearly 129k curated objects, which more than doubles the available amount in prior work on LLM-backed scene generation [26], is finally divided into train (80%), validation (10%), and test (10%) splits.

We further extend the filtering stages to provide compatibility with procedural house generation via ProcTHOR [21] relying on annotated placement options (prompt used with GPT-4o listed in Fig. 22). Rather than generating new placement options for all synsets, we rely on the already diverse set of THOR object categories and map each synset to the most relevant THOR category, if any, using scale and semantic similarity-based heuristics. In more detail, for each candidate pairing, we compute a weighted dissimilarity score based on room-type compatibility, feasible placement locations, object scale ranges, boolean affordances, and calibrated WordNet semantic similarity. Candidate matches are filtered using hard constraints on room and scale compatibility, followed by a lightweight reranking stage that emphasizes semantic alignment. Each synset is assigned to the lowest-cost THOR category below a fixed threshold, yielding a robust many-to-one alignment suitable for downstream scene generation. We then let the procedural engine sample objects from any synset associated with the currently selected THOR category. Using these heuristic placement reference assignments, the additional filtering stages ensure that objects (1) have synsets with valid placement references,

- (2) have annotated pickable/receptacle properties compatible with those of the placement reference, (3) have bounding box scales suitable for placement in houses, (4) are sufficiently shallow if wall-placeable, (5) have synsets that are not hyponyms of weapons, and (6) have synsets present in the train split. The result of applying these additional filters is a curated subset of 92.5k objects, corresponding to ∼2k synsets.

For all objects, we provide an extensive metadata collection that includes the physical parameters of scale and mass as well as semantic information: name, category, and synsets. To enable the easy use of these models in robotics simulations, we also provide convex meshes and grasps. In addition to this, we make sure that these objects are defined in a canonical coordinate system to make them easily loadable into scenes.

###### B MolmoSpaces-Scenes-MultiType Generation Details

Grid layout placement option As mentioned in Sec. 3.1, we extend the original DFS-based floor object solver in Holodeck [26] with a ‘grid’ global constraint that enables structured batch placement of multiple objects suitable for several non-residential MSMultiType scene types. Objects to be placed with a grid constraint are handled jointly: the solver enumerates feasible grid shapes (rows × columns) that can accommodate an iteratively decreasing object set –starting with the original object count requested by the LLM–, prioritizing compact layouts. For each candidate grid shape, the solver attempts to place the entire grid footprint within the current room and greedily assigns objects to grid cells by selecting positions and rotations that maximize satisfaction of remaining (non-grid) constraints regarding already placed objects. Grid configurations are scored by aggregating per-object constraint satisfaction, including an additional soft bias discouraging obstruction of door-to-door circulation. The highest-scoring grid placement is retained before resuming DFS over the remaining objects. We optionally apply a small positional and rotational jitter to grid-constrained objects, retaining only collision-free perturbations. Fig. 18 illustrates how the new grid layout modality can simplify the task of placing uniformly spaced objects.

Style accents with persona descriptions Persona descriptions conveying stylistic information were filtered from the full set in PersonaHub [42] via GPT-4.1-mini prompted as shown in Fig. 23. The effects of adding samples from this filtered set to a scene specification via the simple binding structure used to generate the ∼110k scenes in MSMultiType are illustrated by Fig. 19.

Original Holodeck [26] prompt No edge placement emphasis New prompt with grid layout

... + Using a grid constraint helps me create more realistic layouts when placing multiple objects in a repetitive pattern across a floor area that are meant to be away from room edges. Since grids are likely to occupy a considerable amount of space, it is convenient to place them as early as possible after the anchor object...

... edge: at the edge of the room, close to the wall

... edge: at the edge of the room, close to the wall, most of the objects are placed here... I prefer objects to be placed at the edge (the most important constraint) of the room if possible, which makes the room look more spacious...

- , most of the objects are placed here... I prefer objects to be placed at the edge (the most important constraint) of the room if possible, which makes the room look more spacious...

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

- Figure 18 Object placement versus inclusion of the ‘grid’ layout constraint. In all samples, the textual scene specification in the LLM prompt is ‘a primary school classroom’. In the bottom row we additionally disable a bias term meant to improve door-to-door navigability in multi-room scenes. Text in red (green) indicates removal (addition).

a waiting room

the favorite waiting room of a person who identifies as ‘A vintage vinyl record collector who is challenged to keep their growing collection in check‘

the favorite waiting room of a person that identifies as ‘A curator specializing in health and science exhibits, constantly seeking the epidemiologist’s input to ensure accuracy and educational value‘

the favorite waiting room of a person who identifies as ‘A former PBA basketball player who still holds a grudge against the alumni for leaving the team‘

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

- Figure 19 Complementing scene specifications with persona descriptions produces notable stylistic changes and inclusion of a wider variety of object types in LLM-generated scenes.

SYSTEM PROMPT:

You are an expert in 3D model analysis. Your main task is to identify 3D models that contain more than a single object (e.g., kitchens with cabinets and sinks, dining tables with chairs, or assortments of objects of some or multiple categories, to name a few examples). You also decide whether the 3D model seems to have missing geometry, e.g. resulting from an incomplete scan, or excessive geometry, e.g., resulting from a scan that also reconstructs part of the supporting surface or background, or due to additional props added as decoration for rendering. Among others, a model without excessive geometry should not make the main object appear as being mounted or placed on any type of surface or structure. Finally, you also decide whether the model’s object contains surfaces that can be used as receptacles of smaller object types (like seats of a chair or sofa, or top of a table, among others), and whether the model seems realistically textured for an object of the given category.

The provided data is a collage of renders of the 3D model from different perspectives, showing the front, left, back, and right sides. The expected background should be a flat white color, with everything else being part of the model.

Your decisions should be structured as a JSON dict with the following entries:

- - ‘object types’: List[str], with each entry corresponding to an object type present in the model, e.g., [’shoe’] for a model of a pair of shoes;
- - ‘object instances’: Dict[str, int], with an approximate count of instances of each type, e.g., ’book’: 3 for a model of a pile of 3 books);
- - ‘main object type’: str, type of either largest object, or the most meaningful one, e.g., ’plate’ for a model of a plate with a fork on top;
- - ‘supporting structures’: List[str], typically empty, with names of visible structures supporting the main object, e.g. fragments (or panels) of tables, walls, floors, rugs, etc.;
- - ‘excessive geometry’: str, typically ‘No excessive geometry’, briefly describing the presence of supporting or background surfaces possibly added to the model as decoration for rendering or due to poor segmentation of a reconstructed or scanned model;
- - ‘has excessive geometry’: bool, providing a binary response from the analysis under ‘excessive geometry’;
- - ‘detachable part types’: List[str], typically empty, including easily detachable parts of the main object that should not be considered additional objects as they appear in place, e.g., [‘light bulb’, ‘lamp shade’] for a model of a table lamp with those parts visible and mounted at their proper places on the main object;
- - ‘is single object’: bool, considering all previous responses, whether the model only contains one object (no additional objects, other than possibly detachable parts that appear properly installed on the object) and does not include excessive geometry of any kind;

...

- Figure 20 System prompt used with GPT-4.1 to generate alternative annotation for Objaverse object assets filtering in batch mode (continues in Fig.21).

Table 8 Comparison of large-scale grasp datasets. MolmoSpaces provides the highest number of object instances.

Dataset Year # Objects # Grasps

HO-3D 2020 10 78,000 EGAD 2020 2,331 233,000 DDG 2020 500 50,000 DexYCB 2021 20 582,000 Acronym 2021 8,872 17,000,000 UniGrasp 2020 1,000 2,000,000 DexGraspNet 2023 5,355 1,300,000 Fast-Grasp’D 2023 2,350 1,000,000 GenDexGrasp 2023 58 436,000 MultiGripperGrasp 2024 345 30,400,000 GraspGen 2025 8,515 53,100,000

MolmoSpaces-Grasps (Ours) 2026 48,675 42,000,000

...

- - ‘missing geometry’: str, typically ‘No missing geometry’, briefly describing where the model appears to have missing geometry, e.g., no geometry around the back side, resulting in empty render or some distorted view of the front one for that perspective. Do not take into consideration whether the model appears to be lacking textures;
- - ‘has missing geometry’: bool, providing a binary response from the analysis under ‘missing geometry’;
- - ‘receptacles’: List[str], generally empty for small objects, including names of the main object’s surfaces that can be used as receptacles for smaller objects, as their normals are oriented upward and have sufficient area with no/low curvature, e.g. [‘top of mattress’] for a model of a bed with an installed mattress.
- - ‘texture quality’: int, range 0 (no texture, making the object in the model hard to identify) to 9 (detailed and realistic texture for the object type, making the model appear close to real).

An example response to a query with renders of a model depicting a bookshelf with about 10 books in red, blue, and white placed on a green tiled floor where the render from the back shows the same spines seen from the front could be:

<EXAMPLE JSON OMITTED> Feel free to briefly reason before providing you response as a JSON parseable dict, which must include clear and concise values for all the required entries without requesting additional input.

- Figure 21 System prompt used with GPT-4.1 to generate alternative annotation for Objaverse object assets filtering in batch mode (continues from Fig.20).

We need to generate placement options for synsets used to label objects. For each of these synsets, we have associated specific types, text descriptions of some objects, and scale ranges.

Write the placement options in JSON format, and do not add any additional comments. The structure is a mapping from each synset to a nested mapping with Boolean entries, indicating whether each property seems likely for objects of the given synset: {

‘hasMultipleObjects’: refers to a composition or set of several objects, ‘isScene’: refers to a scene or a set of objects (like an assortment) rather than a single object, ‘roomTypes’: {

‘inKitchens’: appears in a kitchen, ‘inLivingRooms’: appears in a living room, media room, or dining room, ‘inBedrooms’: appears in a bedroom, office, or playroom, ‘inBathrooms’: appears in a bathroom, restroom, or laundry room, ‘inOthers’: appears in other room types like garages, balconies, etc.

}, ‘feasibleLocations’: { ‘onFloorInCorner’: placed directly on the floor, in a corner, ‘onFloorInMiddle’: placed directly on the floor, anywhere away from walls, ‘onFloorOnEdge’: placed directly on the floor and in contact with a wall, ‘onWall’: placed on a wall, ‘fromCeiling’: hangs from the ceiling }, ‘isPickupable’: allows being picked up with a single hand, ‘isKinematic’: has an effective fixed pose, as in Unity’s kinematic bodies, ‘multiplePerRoom’: appears multiple times in the same room

}

Please set at least one of the ‘roomTypes’ (try to use ‘ìnOthers’ sparingly) unless the synset seems to refer to a scene rather than a single object as signaled in ‘ìsScene’. Also note that enabling any of the ‘feasibleLocations’ options (floor, wall, or ceiling) prevents the objects os the synset from being placed on top of other structures or furniture.

The synsets (along with associated specific types, sample descriptions, and scale ranges in cm) to annotate are:

<OMITTED BATCH OF INPUT DATA>

- Figure 22 Prompt used with GPT-4o to determine placement options for a given batch of synsets.

SYSTEM PROMPT:

You are an expert in sociology, psychology, and interior design counseling. Given a list of one-line personal identity statements, your task is to identify which individuals are likely to find certain indoor scenes visually memorable due to the presence of large, interest-relevant objects. These objects should stand out even in visually cluttered environments.

Typically, only about 25% of the identities will be sufficiently specific or interest-oriented to warrant this kind of visual sensitivity. Avoid selecting identities that are too abstract, ethnicity-related, enterprise-centric, or related solely to virtual environments, as these are unlikely to correspond to specific, visually memorable physical elements.

Return your output as a JSON-parseable dict mapping indices corresponding to identity statements that are useful for informing visually impactful interior design styles to a string describing visual objects, decoration items, pieces of furniture, etc., that would instantly draw the attention of each chosen personality in some indoor scene.

- Figure 23 System prompt used with GPT-4.1-mini to select valid persona descriptions in batch mode.

100%

80%

60%

40%

20%

0%

Creditcard

Butterknife

Plunger

Spatula

Pen

Potato

Salt

Mug

Bottle

SaltShaker

Egg

Remote

Fork

Alarm

WineBottle

VaseDecorative

Cup

Ladle

PepperShaker

VaseMedium

VaseTall

TissueBox

PaperTowel

Bowl

VaseFlat

SprayBottle

WavingStatue

SoapBottle

Apple

FertilityStatue

Keychain

Spoon

TabletopDecor

Butter

StoneStatue

DogStatue

Kettle

Pencil

Watch

Tomato

Knife

TowelStatue

Cellphone

Candle

HandStatue

Cloth

Pan

Cd_1

Plate

TennisRacquet

Pot

Book

AlarmClock

VaseOpen

(a) Average grasp success rates by object category in a random sample of MolmoSpaces houses.

100%

80%

60%

40%

20%

0%

Drawer

Desk

SideTable

Cabinet

Laptop

Dishwasher

Fridge

Oven

Showerdoor

Box

Safe

LaundryHamper

Microwave

(b) Average grasp success rates by category for articulated objects in a random sample of MolmoSpaces houses.

- Figure 24 Average grasp success rates by category for all objects tested in a random sample of scenes from MSCrafted and MSProcObja MolmoSpaces-Scene dataset.

