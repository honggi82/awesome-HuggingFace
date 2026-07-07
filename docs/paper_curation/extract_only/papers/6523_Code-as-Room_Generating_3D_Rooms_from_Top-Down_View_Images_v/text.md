## Code-as-Room: Generating 3D Rooms from Top-Down View Images via Agentic Code Synthesis

#### Yixuan Yang1∗ Zhen Luo2,3∗ Wanshui Gan1∗ Jinkun Hao1

Junru Lu4 Jinghao Yan1 Zhaoyang Lyu1 Xudong Xu1† 1 Shanghai Artificial Intelligence Laboratory 2 Shanghai Innovation Institute 3 Southern University of Science and Technology 4 University of Warwick

# arXiv:2605.18451v1[cs.CV]18May2026

Project page: https://code-as-room.github.io/

[Figure 1]

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

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

Input Any Top-Down view Room Images Output 3D Scenes in Blender

Figure 1: Code-as-Room brings diverse interactive 3D scenes from a single top-down view image. We design an agentic system with a structured execution harness and activate the MLLMs’ ability to understand, design, and code the 3D rooms in Blender.

### Abstract

Designing realistic and functional 3D indoor rooms is essential for a wide range of applications, including interior design, virtual reality, gaming, and embodied AI. While recent MLLM-based approaches have shown great potential for 3D room synthesis from textual descriptions or reference images, text-based methods struggle to capture precise spatial information, and existing image-conditioned agents suffer from instability and infinite looping when tasked with holistic room generation from top-down views. To address these limitations, we propose Code-asRoom, an MLLM-based agentic framework equipped with a structured execution harness, which represents 3D rooms with Blender codes. Given a top-down room image, the framework parses the reference image to extract scene elements and their spatial relationships, and synthesizes executable Blender code for geometry, materials, and lighting in a principled, multi-stage pipeline. A cross-stage memory module is maintained throughout to mitigate context forgetting inherent to existing agent-based frameworks. We further introduce a dedicated benchmark for codebased 3D room synthesis, encompassing various evaluation protocols. Based on our benchmark, comprehensive comparisons against existing agent-based methods are conducted to validate the effectiveness of our proposed execution harness.

*Equal contribution. †Corresponding author. ‡Contact: arnoldyang97@gmail.com.

Preprint. Under review.

### 1 Introduction

Designing realistic and functional 3D indoor rooms plays a crucial role in the interior design, virtual reality, games, and even embodied AI [2, 21, 12, 11]. However, manually constructing 3D rooms is labor-intensive, requiring expertise in 3D object modeling, spatial arrangement, material design, and light adjustment. Traditional graphics methods have explored procedural generation, rule-based layout optimization, and constraint-driven object placement to reduce manual effort in room creation [26, 5, 3, 33, 17, 34]. Admittedly, these methods are severely limited by hand-crafted rules and predefined categories, failing to handle complex real-world spatial relationships and flexible user needs.

Recently, multimodal large language models (MLLMs) [18, 9] have experienced significant prosperity, demonstrating remarkable performance and strong generalization across various application scenarios. Thus, leveraging the power of MLLMs for 3D room generation has become an intriguing research problem. A line of scene generation approaches [6, 30, 4, 7, 28, 23, 29, 16] represents indoor scenes as JavaScript Object Notation (JSON), or other structured formats, and resorts to MLLMs to predict spatial layout, inter-object relations, and object attributes according to users’ scene descriptions. Afterwards, 3D objects are retrieved or generated based on their attributes to compose a complete 3D room. Nevertheless, text descriptions of scenes fall short of specifying interior spatial information, such as object counts, precise locations, or detailed orientations, resulting in generated 3D rooms that fail to align with users’ preferences.

In real-world design workflows, top-down layouts, sketches, and floor-plan-like images are widely employed to facilitate 3D room creation, since they inherently encode rich spatial priors and holistic scene appearance. In practice, 3D designers iteratively refine their work by consulting these references until a satisfactory result is achieved. Following this paradigm, researchers [27, 32, 25, 19, 24] have explored leveraging MLLM agents to synthesize 3D rooms from reference images, where the agent alternates between a generator role and a critic role in an iterative manner. Notably, a seminal work, VIGA [32], proposes an intriguing coding agent that generates executable Blender code to construct 3D scenes, encompassing both the scene layout and the constituent 3D objects. Given a perspective image as input, VIGA reconstructs the corresponding 3D scene via an analysis-bysynthesis loop, demonstrating promising potential for the code-as-scene paradigm. However, when naively extending from perspective images to top-down views for complete 3D room generation, VIGA struggles to recover fine-grained spatial details. More critically, the agent is prone to falling into infinite loops, resulting in unstable and unreliable generation outcomes.

To circumvent this hurdle, we propose Code-as-Room (CaR), an MLLM-based agentic framework equipped with a structured execution harness for top-down image-based 3D room synthesis, which generates executable Blender code to represent 3D rooms. The framework first parses the top-down reference image to identify three categories of scene elements: major furniture, small accessory items attached to them, and interior finishes comprising doors, windows, and walls. Their spatial interrelations are inferred from the image to form a holistic scene graph representation. The framework then generates layout code for all identified elements and iteratively refines the overall spatial arrangement through a render-and-compare feedback loop, empowered by the visual recognition capabilities of the MLLM. Building upon the refined layout, the MLLM is further employed to profile each 3D object by inferring detailed properties including appearance, functional attributes, and material characteristics. Following object profiling, the framework enters the object code generation phase, in which the agent synthesizes Blender code for both geometry and surface materials, strictly conditioned on the inferred object profiles. Asset retrieval and 3D object generation are incorporated as auxiliary modules to handle challenging small items with complex geometric details. Finally, texture, material, and lighting codes are generated to accomplish interior decoration, substantially enhancing the perceptual aesthetics and photorealistic fidelity of the synthesized room. Notably, a cross-stage memory module is maintained throughout the entire pipeline to store the outputs of each stage, effectively mitigating the pervasive context forgetting problem inherent to existing agent-based frameworks.

To systematically evaluate existing MLLM models and our agentic framework, we construct a dedicated benchmark for code-based 3D Room synthesis. Beyond assessing visual quality, the benchmark is designed to evaluate the distinctive challenges of this task, encompassing visual content understanding, spatial relationship reasoning, and vision-to-code generation capability. Moreover, comprehensive comparisons against existing agent-based methods validate the effectiveness of our

proposed harness for code-based 3D room generation. Overall, our contributions can be summarized as follows:

- • We propose a top-down image-guided paradigm for 3D room synthesis, where the input image serves as a global spatial prior to guide complete indoor room generation.
- • We propose a structured execution harness that orchestrates the MLLM agent for codebased 3D room generation, ensuring stable and coherent 3D room synthesis.
- • We introduce an Image-to-3D Room synthesis benchmark and conduct comprehensive experiments to evaluate the existing MLLMs in terms of visual understanding, spatial reasoning, vision-to-code ability, and scene quality.

### 2 Related Work

Procedural and Data-driven Indoor Scene Synthesis Indoor scene synthesis has long been studied in computer graphics. Early methods typically formulate scene generation as a rule-based, constraint-based [17], or optimization-driven problem [33]. For example, constraint-based placement systems allow users to compose complex scenes by specifying semantic and geometric constraints, while furniture layout methods incorporate interior design guidelines, ergonomic objectives, and spatial priors to optimize plausible object arrangements. Other works learn arrangement priors from example scenes, synthesizing new 3D object configurations by modeling object co-occurrence, support relations, and spatial distributions. Beyond major furniture layout, interactive tools such as ClutterPalette [34] further support the placement of small-scale objects to enrich indoor scenes. More recently, procedural generation frameworks such as ProcTHOR [5] have scaled indoor environment construction to large numbers of interactive houses for embodied AI training and evaluation. However, these methods are mostly driven by rules, constraints, examples, or procedural templates, leaving the problem of generating a complete, editable, and executable 3D room from a room-level visual input largely unexplored.

LLM- and Agent-based 3D Scene Generation Large language models have recently been used for 3D scene generation due to their commonsense reasoning and open-vocabulary planning ability. Methods such as Holodeck [30], LLplace [28], LAYOUTVLM [23] and I-Design [4] generate room layouts, object selections, spatial relations, or scene graphs from language instructions, demonstrating the potential of LLMs for controllable indoor scene synthesis. Building on this direction, recent agentic frameworks introduce tool use, feedback loops, and multi-agent collaboration. For example, SceneWeaver [27] employs a self-reflective agent to coordinate different generation tools and iteratively refine scenes, SAGE [25] couples scene generators with critics to produce simulation-ready environments for embodied AI, and SceneSmith [19] uses hierarchical VLM agents to progressively construct indoor scenes from architectural layout to furniture and small objects. Despite these advances, existing LLM- and agent-based methods are still primarily text- or task-driven. In practical design scenarios, however, users often start from floor plans, top-down sketches, or layout images, where room structure and object arrangements are already visually specified. This makes room-level image input a more direct and valuable condition for 3D scene generation, yet it remains underexplored in existing agentic frameworks.

Image-conditioned 3D Generation and Code-based 3D Scene Representation Imageconditioned 3D generation has recently achieved impressive progress, with many methods generating 3D assets from images using diffusion models, neural fields, meshes, or other learned representations [13, 14, 31]. However, most of these methods mainly focus on single objects or relatively simple scenes, and their outputs are often designed for reconstruction or visual synthesis rather than the structured generation of a complete indoor room. As a result, they are not well-suited for modeling room-scale scene elements such as global layout, major furniture, minor objects, material appearance, and lighting in a unified way. A particularly relevant work is VIGA [32], which demonstrates the potential of using code to represent 3D structure from visual input. However, its generation pipeline with inadequate harness design and does not address the synthesis of an entire room with complex vision input. More broadly, recent code-based 3D generation methods [22, 10, 15] have shown that executable programs are a promising representation for 3D content due to their interpretability and editability. Nevertheless, these approaches mainly focus on individual objects or localized structures, rather than representing a complete room-scale scene in code. In contrast,

[Figure 17]

- View1
- View2

Stage5

- Stage3 Generate major objects layout code.

[Figure 18]

[Figure 19]

Visionfeedback

[Figure 20]

Wrong Correct

Vision-feedback loop

- Stage4 Add minor objects layout code.

Parse layout code and image to extract each object‘s detailed properties.

Layout-Grounded Object Profiling

[Figure 21]

Input Image

Stage6

[Figure 22]

[Figure 23]

[Figure 24]

Object geometry code generation.

[Figure 25]

[Figure 26]

Vision-feedback loop

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

Vision-feedback loop

Post-hocVision-feedbackOptimizationloop

- Stage1

Analysis Zone, Architecture, Object Properties.

- Stage2

Generate the major-object Scene Graph

Store the minor-object sidecar

Image-based Scene Structuring

Cross-Stage

Memorysystem Image Stage1 Stage2 Stage3 Stage4 Stage5 …… Stage9 Stage10

[Figure 31]

[Figure 32]

[Figure 33]

- Figure 2: Overview of the Code-as-Room pipeline. A single top-down view image is progressively transformed into a fully renderable 3D scene through a sequence of specialized MLLM agent stages, organized into five phases: image-based scene structuring, layout code generation, layout-grounded object profiling, object-level code generation, and interior decoration code generation. Arrows denote data flow through the cross-stage memory system, wherein each stage reads upstream outputs and writes its own results as typed memory entries.

our work aims to generate a complete indoor room from a top-down image, where the full scene is represented as executable Blender code.

- 3 Method

Stage8,9,10 Per-Part PBR material code gen.

Stage7

[Figure 34]

Enrich the room with code-retrieved objects.

[Figure 35]

Room texture code generation

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

Vision-feedback loop

Vision-feedback loop

[Figure 41]

Lighting code generation.

[Figure 42]

Layout Code Generation

Object-level Code Generation

Interior Decoration Code Generation

Coarse VL Understanding

#### 3.1 Problem Definition

Given a room-level top-down image I, our goal is to generate executable Blender code C that constructs a complete 3D indoor scene aligned with the input image. The code specifies room structure, object placement, object geometry, materials, lighting. We formulate the task as an agentic imageto-code generation process:

C = A(I), where A denotes the proposed VLM-agent harness.

Directly generating complete scene code from a single image is challenging because the model must jointly infer room structure, spatial layout, object geometry, and appearance. We therefore decompose the process into a coarse-to-fine workflow. The coarse stage builds a structured scene state and a layout program:

DCU = UCU(I,M), Clayout = GCG(I,DCU,M),

where DCU denotes the coarse scene understanding result and Clayout is the executable layout code. The fine stage then enriches the layout with image-grounded object descriptions and synthesizes the final room program:

DFU = UFU(I,Clayout,M), C = GFG(I,Clayout,DFU,M).

Here, M is the cross-stage memory shared by all modules. This decomposition separates global spatial alignment from local detail synthesis: the coarse stage fixes room structure and object placement, while the fine stage adds editable geometry, appearance, and rendering code. The whole pipeline of Code-as-Room is shown in Fig. 2.

#### 3.2 Cross-Stage Memory

We maintain a shared memory M as the persistent state of the pipeline. After stage s produces a typed artifact es = ⟨s,τs,Os,ηs⟩, memory is updated by

Ms = Ms−1 ⊕ es.

Each downstream stage reads only a predefined memory view, preserving cross-stage consistency while reducing prompt noise and hallucinated dependencies.

#### 3.3 Image-based Scene Structuring

The first two stages convert the top-down image I into a structured scene state for layout generation. Stage 1 extracts a schema-constrained description D1, and Stage 2 builds an object-centric scene graph G = (V,E) with a minor-object sidecar Mminor.

- Stage 1: Spatial semantic analysis. The VLM outputs D1 with functional zones, object hierarchies, and architectural elements. Each object is assigned an identifier, category, placement type, and parent when applicable, while walls, doors, windows, openings, and built-in structures are kept as fixed spatial references. A perimeter-aware prompt scans walls, corners, openings, and a coarse grid to recover peripheral and wall-mounted objects. The resulting description is

D1 = F1(I,P1), M = M ⊕ D1.

- Stage 2: Object-centric scene graph construction. Stage 2 reads {I,D1} and first derives a deterministic skeleton

S = {Varch,Vmajor,Eparent,Mminor},

where Varch contains architectural features, Vmajor contains layout-defining objects, Eparent contains hierarchy-derived relations, and Mminor stores minor objects for later placement. The VLM only completes attributes, geometry hints, and forward relations among existing nodes. After filtering invalid edges and adding inverse and architectural-anchor relations, the graph and sidecar are written to memory:

E = Eparent ∪ Evlm ∪ Ewall ∪ Ecorner ∪ Inv(·), M = M ⊕ {G,Mminor}.

3.4 Layout Code Generation

Given D1, G = (V,E), Mminor, and I, this stage generates a coarse layout program Clayout. Objects are instantiated as named bounding-box proxies in Blender with approximate position, scale, and orientation, while detailed geometry, materials, lighting, and tiny objects are deferred. Because each placement is emitted as a primitive-constructor call, Clayout can be rendered for feedback and parsed by later stages.

We use two sub-stages. The major sub-stage generates the floor-level arrangement of layout-defining furniture from D1 and G. The auxiliary sub-stage freezes the major layout and appends wallmounted objects D1wall = {o ∈ D1 | o.placement type = wall} and visually salient minor objects from Mminor.

- Stage 3: Major layout with visual feedback. We refine the major layout through a render– critique–revise loop initialized by

C(0) = Generate(I,D1,G).

At each iteration, the current code is rendered into a top-down image and evaluated by a VLM-based critic:

R(t) = Render(C(t−1)),

(A(t),st) = Critique(I,R(t),G), A(t) = Sanitize(A(t),D1,G), C(t) = Revise(C(t−1), A(t)).

Here, st denotes the VLM-assessed layout quality score, which summarizes object coverage, overlap, boundary consistency, and spatial relation correctness based on the rendered view. The critic also outputs textual feedback A(t) describing missing objects, overlaps, boundary violations, and relation errors. Since the critic may occasionally suggest unsupported architectural changes, we sanitize its feedback with respect to D1 and G before revising the code. The loop terminates once st ≥ s⋆ or the maximum number of iterations Tmax is reached, where we set Tmax = 5 in our experiments. The final output is denoted as Clayoutmajor.

- Stage 4: Auxiliary layout for walls and salient minors. Stage 4 complements the major layout with wall-mounted objects and visually salient minor objects. Wall-mounted objects are aligned to the inferred wall planes according to their semantic anchors in the scene graph. For minor objects, we only keep items that are visible and layout-relevant at the coarse scene scale:

Mminor⋆ = {m ∈ Mminor},

where m is visible at the coarse layout scale and not surface-bound. These objects include rugs, floor lamps, plants, and large decorations. Tiny surface-bound objects, such as books, cups, and small tabletop items, are deferred to the later fine-grained placement stage, where they are placed according to supporting surfaces and the memory from Stage 2. The selected wall and minor objects are appended as primitive-constructor calls:

Clayout = Append Clayoutmajor,M⋆ , M ← M ⊕ Clayout.

The resulting layout serves as the scaffold for fine-grained description, geometry generation, and small-object placement.

3.5 Object-level Code Generation

After layout code is fixed, the object-level module enriches each proxy with image-grounded appearance, procedural part geometry, and surface-bound small objects.

- Stage 5: Layout-grounded object description. The coarse layout fixes each object’s identifier, category, pose, and size, but lacks visual details for geometry, materials, and textures. We parse Clayout into placed objects and use their layout attributes to ground the VLM. Conditioned on the input image and memory, the VLM produces fine-grained object descriptions DFU covering color, material, function, structure, and style, together with a global room-style description JSON file sroom. These outputs preserve fixed placement and are written to memory.
- Stage 6: Object geometry replacement. For each placed object oi in Clayout, the geometry agent predicts a semantic 3D geometry primitive decomposition:

##### Pi = Φgeo(oi,di) = {pi,j}K

- i
- j=1,

where di ∈ DFU, and each part pi,j specifies a 3D geometry primitive type, semantic part name, local size, offset, and rotation. Since parts are defined in the local frame of the original proxy, the generated object inherits the coarse-layout pose. We replace proxy constructors in the layout program with part-based constructors:

##### Cgeom = Replace Clayout,{oi  → Pi}Ni=1 .

The same mapping is stored as a geometry dictionary for later surface discovery and material assignment.

Tiny objects are instantiated through a hybrid procedural-and-retrieval strategy. For visually distinctive objects, we first create simple geometric placeholders to preserve their positions and occupied regions, and then retrieve matching assets from an asset library B to replace these placeholders. The selected asset is obtained by

b⋆ = arg max

match(b;label,description,placeholder size),

b∈B

where the matching score jointly considers semantic relevance and size compatibility. The selected asset is scaled and aligned to the corresponding placeholder, preserving its support surface and footprint. And the surface discovery and occupancy detection algorithm can be found in the Appendix.

#### 3.6 Interior Decoration Code Generation

After object-level generation, we complete appearance and illumination through geometrypreserving code rewriting:

Cobj −−−−−−→ApplyMat Cmat −−−−−−→ApplyTex Ctex −−−−−−−−→RenderSetup Craw, where each step rewrites or appends Blender code without modifying object placement or geometry.

- Stage 8: Material assignment. Since objects are decomposed into semantic parts, we assign part-level PBR materials using the part dictionary, fine-grained descriptions, and global room style. The agent predicts material type, linear-RGB base color, roughness, metallic value, and specular strength, which are injected into the Blender script and bound to part constructors. Glass and mirrorlike surfaces use shader overrides triggered by material type or part-name keywords; floors and walls use procedural shader nodes.
- Stage 9 : Texture and decorative surfaces. For large or patterned surfaces, we use a highcapacity image generation model to synthesize texture maps for floors, walls, rugs, paintings, posters, and decorative panels, and inject them into the scene by augmenting the corresponding material node graphs with image-texture nodes. Planar decorative elements are assigned explicit UV mappings to ensure proper texture placement, while failed generations fall back to simplified prompts for more robust synthesis.
- Stage 10: Lighting, rendering, and post-hoc correction. In the final stage, we complete the scene by setting up lighting, rendering parameters, and deterministic post-hoc corrections. Conditioned on the input image and the generated scene, the VLM infers the overall lighting style, including the dominant illumination direction, window-driven natural light, possible artificial light sources, and ambient intensity. These cues are then translated into Blender light objects and renderer

settings to produce the raw executable scene Craw. Before final rendering, we apply a deterministic correction pass that improves robustness without changing the semantic layout. This pass fixes common implementation issues such as missing material assignments, invalid texture paths, unreasonable light intensities, incomplete camera coverage, and geometric artifacts.

For movable objects with boundary or overlap violations, we search for a nearby feasible placement:

x⋆i = arg min

∥x − xˆi∥2 s.t. B(oi,x) ⊆ Broom, B(oi,x) ∩ B(oj) = ∅.

x∈N(xˆi)

Here, xˆi is the generated position, N(xˆi) is a local grid neighborhood, Broom is the room boundary, and the collision constraint is applied to nearby non-parent objects. In practice, this projection is implemented by deterministic local search with boundary clamping and stacking offsets for supported objects. The final program is C = PostHoc(Craw), where the final code C serves as the complete representation of the generated 3D room scene, and can be directly executed in Blender to instantiate the full scene.

### 4 Experiments

#### 4.1 Experimental Setup

We mainly conduct two evaluations. First, we propose a benchmark to evaluate the effectiveness of our agentic harness with different VLMs, as well as the performance of different VLMs on this

- Table 1: Our proposed benchmark evaluates different vision-language models for code-based 3D room generation from the top-down reference image, across various metrics, including visual content understanding, spatial reasoning, vision-to-code generation, and holistic scene quality.

VLM Visual Understanding Spatial Reasoning Code Generation Scene Quality

Obj. Recall ↑

Func. Acc. ↑

Self Overlap ↓

Layout IoU ↑

Spatial Relation ↑

Rotation Acc. ↑

Support Acc. ↑

Agent Completion ↑

Exec. Rate ↑

Image Similarity ↑

Scene Usability ↑

Aesthetic Quality ↑

Gemini3.1-pro [9] 17.8% 15.3% 8.4% 16.8% 54.7% 78.0% 41.7% – 57.8% 2.49 3.24 2.51 GPT-5.5 [18] 42.2% 71.7% 14.5% 46.2% 50.8% 65.3% 52.6% – 42.2% 5.8 5.42 7.24

Gemini3-flash w/CaR [8] 58.9% 88.42% 2.57% 72.0% 76.9% 93.5% 93.5% 100% 100% 8.32 6.07 7.17 Gemini3.1-pro w/CaR [9] 55.5% 84.3% 3.3% 73.2% 79.8% 93.6% 94.0% 100% 95.5% 8.08 7.05 8.20 GPT-5.5 w/CaR [18] 67.5% 72.54% 10.5% 66.7% 71.4% 92.2% 80.1% 71.1% 73.3% 7.28 6.00 7.52

task. Second, we compare our method with direct VLM generation and the agentic image-to-3D pipeline VIGA [32]. We further introduce image translation as re-rendering to demonstrate the future potential of this task, followed by a series of ablation studies.

#### 4.2 Benchmark for Top-down view image to 3D Room

Benchmark Models We evaluate Code-as-Room (CaR) with three leading VLM backbones: Gemini-3 Flash [8], Gemini-3.1 Pro [9], and GPT-5.5 [18]. These models are selected for their strong visual understanding, spatial reasoning, and code-generation abilities required for top-down image-to-3D scene synthesis. We exclude Qwen-3.6 [20] and Claude Opus-4.6 [1] from the main benchmark, as their preliminary results were often unusable for this task. To assess the effect of our agentic workflow, we also test Gemini-3.1 Pro and GPT-5.5 under direct generation, where each model generates a complete Blender scene from the input image in a single call.

Benchmark Settings We evaluate our method on a test suite of 41 scenes covering diverse room types, scene complexities, and image styles. The suite includes common residential spaces, such as bedrooms, kitchens, and living rooms, which are grouped into Simple, Middle, and Hard levels according to spatial scale and object density. We further include specialized scenes, such as laboratories, barber shops, and cafes, to test robustness beyond standard home environments. The input images span photorealistic photos, synthetic renderings, and abstract line drawings. Since accurate ground truth is unavailable for such diverse inputs, we build annotations through a human-in-theloop pipeline. Specifically, Gemini 3.1 is first used within our agent workflow to generate coarse labels, which are then corrected by human annotators through a reverse code refinement tool that synchronizes visual edits with the underlying scene code.

Benchmark Metrics We evaluate different VLMs from four complementary aspects: visual understanding, spatial reasoning, code generation, and overall scene quality. For visual understanding, we report object recall and functional accuracy, measuring whether the generated code recovers annotated objects and reconstructs major functional regions. For spatial reasoning, we use self-overlap, layout IoU, spatial-relation consistency, rotation accuracy, and support accuracy to evaluate global layout alignment and local placement correctness. For code generation, we report agent completion rate and Blender execution rate, measuring whether the full multi-stage pipeline can be completed and whether the final code runs successfully in Blender. For overall scene quality, we use VLMbased scores on image similarity , scene usability, and aesthetic quality, together with human-study ratings on usability and error acceptability.

Benchmark Results Table 1 shows that our agentic design consistently improves VLM-based image-to-3D room generation. Direct GPT-5.5 performs strongly in visual understanding and holistic quality, but still suffers from spatial inconsistency and unstable Blender execution. Gemini models are weaker under direct prompting, yet become the most stable and competitive backbones when equipped with Code-as-Room. These results indicate that staged decomposition, memory-guided reasoning, and visual feedback are critical for reliable 3D scene synthesis.

Figure 4 further shows that direct VLM generation often produces incomplete or spatially unstable rooms, whereas Code-as-Room yields more complete structures, clearer functional regions, and better-aligned furniture layouts. The detailed comparisons in Figure 5 show that Gemini-based variants recover richer local structures such as shelves, cabinets, decorations, and tabletop objects,

[Figure 43]

[Figure 44]

[Figure 45]

GPT-5.5 with CaR

Detail

Gemini3-flash with CaR

Gemini3.1-pro with CaR

Detail Detail

Input Top-down view Images

|[Figure 46]|
|---|

|[Figure 47]|
|---|

|[Figure 48]|
|---|

[Figure 49]

|[Figure 50]|
|---|

|[Figure 51]|
|---|

[Figure 52]

|[Figure 53]|
|---|

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

|[Figure 59]|
|---|

|[Figure 60]|
|---|

|[Figure 61]|
|---|

[Figure 62]

[Figure 63]

|[Figure 64]|
|---|

|[Figure 65]|
|---|

[Figure 66]

[Figure 67]

|[Figure 68]|
|---|

- Figure 3: Qualitative comparisons corresponding to the benchmark results in Table 1. With our harness in Code-as-Room, Gemini3-Flash, Gemini3.1-Pro, and GPT-5.5 can all generate executable

- 3D room code from top-down image inputs. The Gemini models achieve more stable results and demonstrate stronger spatial understanding than GPT-5.5, especially in object placement, layout consistency, and spatial relation preservation.

while GPT-5.5 better preserves coarse layouts but tends to simplify small details. Overall, successful image-to-3D room generation depends not only on the base VLM capability, but also on the structured agentic workflow used to organize it.

#### 4.3 Human Evaluation

We conduct a human evaluation with 20 experts, who rate each scene by similarity, usability, lighting alignment, and acceptability. Acceptability measures whether the generated scene can be used after only minor manual corrections. We compare direct VLM generation, Code-as-Room variants, and VIGA [32], with direct Gemini3.1-Pro and GPT-5.5 also serving as the Single-pass LLM baseline.

Table 2: Human evaluation of overall scene quality.

Method Sim. ↑ Use. ↑ Light ↑ Accept. ↑

- (a) Direct Generation Baselines Gemini3.1-Pro / Single-pass [9] 2.0 0.0 4.0 1.0 GPT-5.5 / Single-pass [18] 7.0 6.0 6.5 5.0 VIGA [32] 5.5 4.5 8.0 4.0

- (b) Code-as-Room Variants CaR w/ GPT-5.5 [18] 7.5 7.0 8.0 6.5 CaR w/ Gemini3-Flash [8] 8.5 8.0 8.0 7.5 CaR w/ Gemini3.1-Pro [9] 9.0 8.0 8.0 7.5

Table 2 shows that Code-as-Room consistently improves human-perceived quality over direct generation and VIGA. Our pipeline with Gemini3.1-Pro achieves the best similarity, usability, and acceptability scores. Although VIGA obtains comparable lighting, its lower similarity and usability indicate weaker layout preservation and practical usability. As shown in Figure 6, VIGA tends to generate template-like scenes with missing details and inaccurate object placement. By contrast, Code-as-Room better preserves room proportions, furniture arrangements, and local semantics, demonstrating the benefit of our multistage agentic workflow.

#### 4.4 Scene Re-rendering

To better show the potential of our generated results, we further apply image-level re-rendering to the Blender-rendered scenes. While our 3D scenes are editable and geometrically consistent, many objects are still constructed from primitive shapes, which limits their visual realism. Nevertheless, these scenes provide strong 3D priors, including room structure, object layout, spatial relations, and camera-consistent geometry. We therefore use GPT-5.5 [18] to enhance the rendered images. As shown in Figure 7, with stable geometric guidance from our 3D scenes, the re-rendering model produces more realistic materials, lighting, and object details without additional priors. The enhanced

results also preserve the original layout and maintain geometric and semantic consistency across multiple views, demonstrating that our generated 3D scenes can serve as effective structural priors for high-quality visual refinement.

#### 4.5 Ablation Studies

We conduct ablation studies to analyze the memory mechanism and the visual feedback loop in Code-as-Room. All variants use the same VLM backbone and input images for fair comparison. As shown in Table 3, removing memory degrades all metrics. Without cross-stage memory, later stages cannot reliably reuse earlier imagederived information, leading to missing objects and weaker layout preservation. The drop is especially large in Layout IoU, showing that memory is important for maintaining spatial consistency across stages. For visual feedback, performance improves from 0 to 5 iterations, indicating that intermediate renderings help correct object omissions, placement errors, and rotation misalignment. However, increasing the feedback rounds to 10 reduces Layout IoU and rotation accuracy, suggesting that excessive revision may introduce layout drift or over-correction. Therefore, we use five feedback iterations as a balance between generation quality and refinement cost. Figure 8 provides qualitative examples of these effects.

Table 3: Ablation study on different components of Code-as-Room.

Configuration Obj. Recall ↑ Layout IoU ↑ Rotation Acc. ↑

- (a) Effect of Memory Mechanism w/o Memory 48.2% 58.0% 88.4% Full Model (Ours) 55.5% 73.2% 93.6%

- (b) Effect of Visual Feedback Iterations w/o Visual Feedback (0 iter.) 33.8% 64.0% 71.9% Feedback ×3 35.6% 65.7% 73.2% Feedback ×5 (Ours) 38.4% 66.2% 75.4% Feedback ×10 39.1% 64.2% 72.6%

### 5 Conclusion

In this paper, we presented Code-as-Room, an MLLM-based agentic framework for synthesizing realistic and functional 3D indoor rooms from top-down reference images. Unlike text-driven room generation methods, our framework uses the input image as an explicit spatial prior and represents the generated scene as executable Blender code, enabling editable, renderable, and structured 3D room synthesis. To improve the stability of holistic room generation, Code-as-Room decomposes the task into a principled multi-stage pipeline that progressively performs visual parsing, spatial reasoning, scene layout construction, object-level code generation, and material-lighting refinement. A cross-stage memory module is further introduced to preserve scene information across stages and reduce context forgetting in long agentic workflows. We also built a dedicated benchmark for codebased 3D room synthesis, covering visual understanding, spatial reasoning, code generation, and holistic scene quality. Comprehensive experiments demonstrate that the proposed execution harness substantially improves different MLLM backbones and outperforms existing agent-based baselines in generating coherent, usable, and visually aligned indoor scenes.

Limitations and future work. First, it currently targets global 3D scene synthesis from top-down view images and is not yet optimized for arbitrary-view inputs, which limits its applicability to more general real-world settings. Second, many real-world objects remain difficult to faithfully generate with procedural code, due to the limited alignment between code-generation models and 3D asset. Thus, retrieval-based asset insertion may still be needed for higher geometric fidelity. Second, although image re-rendering can improve visual realism while preserving multi-view geometry and semantics, current video models still struggle with high-quality and temporally consistent rerendering, especially for trajectories longer than five seconds. In future work, we will first explore video generation models as neural renderers for Code-as-Room to produce more realistic and coherent scene visualizations.

Gemini3-flash with CaR

GPT-5.5 Input Gemini3.1-pro with CaR GPT-5.5

Gemini3.1-pro with CaR

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

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

- Figure 4: Representative qualitative results corresponding to the benchmarks presented in Table 1.

[Figure 93]

Input Gemini3.1-pro with CaR

[Figure 94]

Gemini3-flash with CaR

GPT-5.5 with CaR

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

|[Figure 100]|
|---|

|[Figure 101]|
|---|

|[Figure 102]|
|---|

|[Figure 103]|
|---|

|[Figure 104]|
|---|

[Figure 105]

|[Figure 106]|
|---|

Detail Detail Detail

- Figure 5: Detailed qualitative analysis and performance comparisons relative to the benchmarks in Table 1.

[Figure 107]

Input Image CaR with Gemini3.1-pro VIGA

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

- Figure 6: Comparative performance analysis of Gemini 3.1-Pro integrated with CaR (ours) versus the VIGA framework.

Input Image

Top-view

Re-Rendering Details

Rendering Details Re-Rendering

&Rendering Other View Rendering

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

- Figure 7: Visual Enhancement Comparison: From Base 3D Scenes (left) to Realistic Re-rendering by GPT-5.5 (right).

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

Input Image CaR with Memory Sys. CaR without Memory Sys.

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

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

Iteration 0 Iteration 3 Iteration 5 Iteration 10

[Figure 177]

- Figure 8: Ablation study results evaluating the impact of the memory system and visual feedback iterations (referencing data in Table 3).

### References

- [1] Anthropic. System card:claude opus 4.6. Anthropic Technical Report, 2026. URL https://www-cdn. anthropic.com/0dd865075ad3132672ee0ab40b05a53f14cf5288.pdf. Accessed: 2026-05-12.
- [2] Iro Armeni, Ozan Sener, Amir R Zamir, Helen Jiang, Ioannis Brilakis, Martin Fischer, and Silvio Savarese. 3d semantic parsing of large-scale indoor spaces. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 1534–1543, 2016.
- [3] J Timothy Balint and Rafael Bidarra. A generalized semantic representation for procedural generation of rooms. In Proceedings of the 14th International Conference on the Foundations of Digital Games, pages 1–8, 2019.
- [4] Ata C¸elen, Guo Han, Konrad Schindler, Luc Van Gool, Iro Armeni, Anton Obukhov, and Xi Wang. Idesign: Personalized llm interior designer. In European Conference on Computer Vision, pages 217–234. Springer, 2024.
- [5] Matt Deitke, Eli VanderBilt, Alvaro Herrasti, Luca Weihs, Kiana Ehsani, Jordi Salvador, Winson Han, Eric Kolve, Aniruddha Kembhavi, and Roozbeh Mottaghi. Procthor: Large-scale embodied ai using procedural generation. Advances in Neural Information Processing Systems, 35:5982–5994, 2022.
- [6] Weixi Feng, Wanrong Zhu, Tsu-jui Fu, Varun Jampani, Arjun Akula, Xuehai He, Sugato Basu, Xin Eric Wang, and William Yang Wang. Layoutgpt: Compositional visual planning and generation with large language models. Advances in Neural Information Processing Systems, 36:18225–18250, 2023.
- [7] Rao Fu, Zehao Wen, Zichen Liu, and Srinath Sridhar. Anyhome: Open-vocabulary generation of structured and textured 3d homes. In European Conference on Computer Vision, pages 52–70. Springer, 2024.
- [8] Google Gemini Team. Gemini 3.1 flash-lite model card. Google DeepMind Technical Report, 2026. URL https://storage.googleapis.com/deepmind-media/Model-Cards/ Gemini-3-1-Flash-Lite-Model-Card.pdf. Accessed: 2026-05-12.
- [9] Google Gemini Team. Gemini 3.1 pro model card. Google DeepMind Technical Report, 2026. URL https://storage.googleapis.com/deepmind-media/Model-Cards/ Gemini-3-1-Pro-Model-Card.pdf. Accessed: 2026-05-12.
- [10] R Kenny Jones, Theresa Barton, Xianghao Xu, Kai Wang, Ellen Jiang, Paul Guerrero, Niloy J Mitra, and Daniel Ritchie. Shapeassembly: Learning to generate programs for 3d shape structure synthesis. ACM Transactions on Graphics (TOG), 39(6):1–20, 2020.
- [11] Wenbin Li, Sajad Saeedi, John McCormac, Ronald Clark, Dimos Tzoumanikas, Qing Ye, Yuzhong Huang, Rui Tang, and Stefan Leutenegger. Interiornet: Mega-scale multi-sensor photo-realistic indoor scenes dataset. arXiv preprint arXiv:1809.00716, 2018.
- [12] Zhengqin Li, Ting-Wei Yu, Shen Sang, Sarah Wang, Meng Song, Yuhan Liu, Yu-Ying Yeh, Rui Zhu, Nitesh Gundavarapu, Jia Shi, et al. Openrooms: An end-to-end open framework for photorealistic indoor scene datasets. arXiv preprint arXiv:2007.12868, 2020.
- [13] Ruoshi Liu, Rundi Wu, Basile Van Hoorick, Pavel Tokmakov, Sergey Zakharov, and Carl Vondrick. Zero1-to-3: Zero-shot one image to 3d object. In Proceedings of the IEEE/CVF international conference on computer vision, pages 9298–9309, 2023.
- [14] Yuan Liu, Cheng Lin, Zijiao Zeng, Xiaoxiao Long, Lingjie Liu, Taku Komura, and Wenping Wang. Syncdreamer: Generating multiview-consistent images from a single-view image. In International conference on learning representations, volume 2024, pages 27676–27697, 2024.
- [15] Sining Lu, Guan Chen, Nam Anh Dinh, Itai Lang, Ari Holtzman, and Rana Hanocka. Ll3m: Large language 3d modelers. arXiv preprint arXiv:2508.08228, 2025.
- [16] Zhen Luo, Yixuan Yang, Xudong Xu, Jinkun Hao, Zhaoyang Lyu, Feng Zheng, Jiangmiao Pang, and Yanwei Fu. Stable: Simulation-ready tabletop layout generation via a semantics-physics dual system,

2026. URL https://arxiv.org/abs/2605.16137.

- [17] Paul Merrell, Eric Schkufza, Zeyang Li, Maneesh Agrawala, and Vladlen Koltun. Interactive furniture layout using interior design guidelines. ACM transactions on graphics (TOG), 30(4):1–10, 2011.
- [18] OpenAI. Gpt-5.5 model card and system card. OpenAI Technical Report, 2026. URL https: //deploymentsafety.openai.com/gpt-5-5/gpt-5-5.pdf. Accessed: 2026-05-12.

- [19] Nicholas Pfaff, Thomas Cohn, Sergey Zakharov, Rick Cory, and Russ Tedrake. Scenesmith: Agentic generation of simulation-ready indoor scenes. arXiv preprint arXiv:2602.09153, 2026.
- [20] Qwen Team. Qwen3.6-27B: Flagship-level coding in a 27B dense model, April 2026. URL https: //qwen.ai/blog?id=qwen3.6-27b.
- [21] Nathan Silberman, Derek Hoiem, Pushmeet Kohli, and Rob Fergus. Indoor segmentation and support inference from rgbd images. In European conference on computer vision, pages 746–760. Springer, 2012.
- [22] Chunyi Sun, Junlin Han, Weijian Deng, Xinlong Wang, Zishan Qin, and Stephen Gould. 3d-gpt: Procedural 3d modeling with large language models. In 2025 International Conference on 3D Vision (3DV), pages 1253–1263. IEEE, 2025.
- [23] Fan-Yun Sun, Weiyu Liu, Siyi Gu, Dylan Lim, Goutam Bhat, Federico Tombari, Manling Li, Nick Haber, and Jiajun Wu. Layoutvlm: Differentiable optimization of 3d layout via vision-language models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 29469–29478, 2025.
- [24] Fan-Yun Sun, Shengguang Wu, Christian Jacobsen, Thomas Yim, Haoming Zou, Alex Zook, Shangru Li, Yu-Hsin Chou, Ethem Can, Xunlei Wu, et al. 3d-generalist: Self-improving vision-language-action models for crafting 3d worlds. arXiv preprint arXiv:2507.06484, 2025.
- [25] Hongchi Xia, Xuan Li, Zhaoshuo Li, Qianli Ma, Jiashu Xu, Ming-Yu Liu, Yin Cui, Tsung-Yi Lin, WeiChiu Ma, Shenlong Wang, et al. Sage: Scalable agentic 3d scene generation for embodied ai. arXiv preprint arXiv:2602.10116, 2026.
- [26] Ken Xu, James Stewart, and Eugene Fiume. Constraint-based automatic placement for scene composition. In Graphics Interface, volume 2, pages 25–34, 2002.
- [27] Yandan Yang, Baoxiong Jia, Shujie Zhang, and Siyuan Huang. Sceneweaver: All-in-one 3d scene synthesis with an extensible and self-reflective agent. Advances in neural information processing systems, 38:140319–140351, 2026.
- [28] Yixuan Yang, Junru Lu, Zixiang Zhao, Zhen Luo, James JQ Yu, Victor Sanchez, and Feng Zheng. Llplace: The 3d indoor scene layout generation and editing via large language model. arXiv preprint arXiv:2406.03866, 2024.
- [29] Yixuan Yang, Zhen Luo, Tongsheng Ding, Junru Lu, Mingqi Gao, Jinyu Yang, Victor Sanchez, and Feng Zheng. Optiscene: Llm-driven indoor scene layout generation via scaled human-aligned data synthesis and multi-stage preference optimization. Advances in Neural Information Processing Systems, 38:42499– 42529, 2026.
- [30] Yue Yang, Fan-Yun Sun, Luca Weihs, Eli VanderBilt, Alvaro Herrasti, Winson Han, Jiajun Wu, Nick Haber, Ranjay Krishna, Lingjie Liu, Kiana Ehsani, and Eric Kolve. Holodeck: Language guided generation of 3d embodied AI environments. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024.
- [31] Kaixin Yao, Longwen Zhang, Xinhao Yan, Yan Zeng, Qixuan Zhang, Lan Xu, Wei Yang, Jiayuan Gu, and Jingyi Yu. Cast: Component-aligned 3d scene reconstruction from an rgb image. ACM Transactions on Graphics (TOG), 44(4):1–19, 2025.
- [32] Shaofeng Yin, Jiaxin Ge, Zora Zhiruo Wang, Chenyang Wang, Xiuyu Li, Michael J Black, Trevor Darrell, Angjoo Kanazawa, and Haiwen Feng. Vision-as-inverse-graphics agent via interleaved multimodal reasoning. arXiv preprint arXiv:2601.11109, 2026.
- [33] Lap-Fai Yu, Sai Kit Yeung, Chi-Keung Tang, Demetri Terzopoulos, Tony F Chan, and Stanley J Osher. Make it home: Automatic optimization of furniture arrangement. ACM Trans. Graph., 30(4):86, 2011.
- [34] Lap-Fai Yu, Sai-Kit Yeung, and Demetri Terzopoulos. The clutterpalette: An interactive tool for detailing indoor scenes. IEEE transactions on visualization and computer graphics, 22(2):1138–1148, 2015.

