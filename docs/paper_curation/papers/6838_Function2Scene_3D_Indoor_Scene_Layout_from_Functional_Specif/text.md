# arXiv:2605.30819v1[cs.CV]29May2026

## Function2Scene: 3D Indoor Scene Layout from Functional Specifications

RUIQI WANG, Simon Fraser University, Canada QIMIN CHEN, Simon Fraser University, Canada DANIEL RITCHIE, Brown University, USA ANGEL X. CHANG, Simon Fraser University, Canada MANOLIS SAVVA, Simon Fraser University, Canada KAI WANG, Simon Fraser University, Canada and ShanghaiTech University, China HAO ZHANG, Simon Fraser University, Canada

[Figure 1]

[Figure 2]

✓ Appropriate working table setup

✓ Proper table size for 6-8 seats

Business traveler's studio (4m x 5m). Staying a few days per month only. Need a proper desk for morning work with laptop. Kitchen should have enough storage to keep the counter clear. Occasionally a colleague over for dinner

Retired couple's great room (7m x 9m). Need to host dinner regularly for 6-8 guests. Cook stays part of the gathering. Doubles as a quiet fireside retreat for two in winter. Enjoy TV shows

✓ Support two adults seated for conversation or dinner

✓ Accessible and clear floor area for fireplace and seating zone

|[Figure 3]<br><br>[Figure 4]|
|---|

|[Figure 5]<br><br>[Figure 6]|
|---|

|[Figure 7]<br><br>[Figure 8]|
|---|

|[Figure 9]<br><br>[Figure 10]|
|---|

Functional prompt Final layout

Zoom-in Functional prompt Final layout Zoom-in

Fig. 1. We present Function2Scene, a framework for generating 3D indoor layouts from functional specifications. Given a detailed functional specification, our method decompose them into functional design constraints, which are then used to iteratively evaluate and refine a generated scene. Please refer to the supplementary material for the full input prompt and more detailed visualizations.

which activities can happen. A room can be visually and semantically plausible, yet still be a poor design: a desk facing direct window glare, a sofa blocking the path to the door, a wardrobe unreachable by a senior who uses it, or a child’s play area hidden from their caregiver. Function is not a secondary attribute — it is one of the primary reasons behind any layout decision.

Most text-driven 3D indoor scene synthesis methods generate rooms from object-centric prompts, asking what furniture should be placed rather than how the space is used. Yet in real interior design, a layout is judged by how well it supports its occupants, e.g., their activities and physical needs. We introduce Function2Scene, a framework for generating 3D indoor layouts from functional specifications, i.e., natural-language design briefs describing who will use a room and what they need to do there. Given such a specification, our system parses occupant personas and activities, derives a customized set of functional design constraints from a taxonomy of 17 criteria spanning spatial, ergonomic, activity, and environmental considerations, and uses these constraints to guide layout generation. Rather than relying on an LLM to directly produce a final scene, Function2Scene performs iterative evaluation and refinement through a tool-augmented check-and-repair loop, combining geometric measurements, LLM-based contextual reasoning, and VLM-based visual assessment. Experiments on 30 professionally written interior-design cases show that Function2Scene produces layouts that better satisfy functional requirements than recent LLM-based scene synthesis baselines, with our results preferred in 94.3% of pairwise comparisons. Our work reframes text-driven indoor scene synthesis from placing plausible objects to designing spaces that support human use.

The computer graphics community has long recognized and studied this. Early scene generation works encoded design guidelines, such as clearance, grouping, alignment, accessibility, as explicit cost functions [Merrell et al. 2011; Yu et al. 2011], but authoring such constraints required significant expertise and effort, making them hard to adapt to different scenarios. To address this limitation, the field pivoted to data-driven approaches, starting with pairwise spatial priors [Fisher et al. 2012; Yu et al. 2011], and advancing through the deep learning era with increasingly expressive learned priors with convolutional neural networks [Wang et al. 2018], autoregressive transformers [Paschalidou et al. 2021], and denoising diffusion [Tang et al. 2024; Zhai et al. 2023]. These work capture increasingly rich statistical patterns, but in doing so, make functional design knowledge more and more implicit and uninterpretable.

1 Introduction

Entering the foundation model era, LLM-based systems [Çelen et al. 2024; Feng et al. 2025, 2023; Sun et al. 2025a; Yang et al. 2024c] enabled flexible text conditioned scene synthesis. More recently, agentic pipelines also enable iterative scene updates to further improve scene quality [He et al. 2026; Luo et al. 2026; Xia et al. 2026; Zhao et al. 2026]. Leveraging foundational priors, these systems support more open-ended scene generation. However, they largely inherit the learning era’s “implicit" approach: an LLM directly generates objects, transforms relations, and even in agentic setups, the

A furnished room is not only a collection of objects, but a proposal for how a space should be used. Furniture layout determines how people circulate, where they sit and look, what they can reach, and

Authors’ Contact Information: Ruiqi Wang, Simon Fraser University, Burnaby, BC, Canada, ruiqi_w@sfu.ca; Qimin Chen, Simon Fraser University, Burnaby, Canada, qca143@sfu.ca; Daniel Ritchie, Brown University, Providence, Rhode Island, USA, daniel_ritchie@brown.edu; Angel X. Chang, Simon Fraser University, Burnaby, Canada, angelx@sfu.ca; Manolis Savva, Simon Fraser University, Burnaby, Canada, manolis. savva@gmail.com; Kai Wang, Simon Fraser University, Burnaby, Canada and ShanghaiTech University, China, kwang.ether@gmail.com; Hao Zhang, Simon Fraser University, Burnaby, Canada, haoz@sfu.ca.

resultis mostlycheckedonly forvisual quality and physical plausibility. Consistent with their goals, most LLM-based methods consume detailed object-centric prompts, for example, “a bedroom with a queen bed, two nightstands, and a dresser.” They are not targeting more “functionality"-oriented prompts, for example, “a bedroom for a couple where one partner reads late while the other sleeps early.” Yet, the latter describes a far more common starting point in real design practices [Kilmer and Kilmer 2024; Panero 1962]. The irony is that LLMs are well suited to the two tasks that limited classical rule-based layout generation in the first place: parsing open-ended functional descriptions, and optimizing functional criteria that are cost prohibitive to manually specify.

We study 3D indoor scene layout from functional specifications: natural-language design briefs that describe who will use a space, what they will do in it, and what constraints their needs impose. Such briefs are closer to an interior design program than to a short text-to-scene command. For the scope of this work, we use professionally written room descriptions adapted from sources such as Architectural Digest. This problem formulation introduces unique challenges, as functional specifications are inherently high-level: instead of directly specifying objects and layout, they impose diverse and heterogeneous constraints over the indoor space: spatial constraints, ergonomic rules, activity patterns, environmental contexts, etc. Consequently, LLMs tasked to directly generate scenes from such functional specification not only struggle with producing scenes that meet the functional demands, but also frequently fail to produce meaningful scenes, without the availability of more explicitly worded prompts.

To address these challenges, we introduce Function2Scene, a framework that empowers LLMs for such functional scenarios. Given a functional specification, we begin by analyzing the “personas," i.e., who the occupants are and what specific needs they have, and the “activities," i.e., what the occupants do in the space. We then generate a detailed scene description, as well as a set of design guidelines drawn from a taxonomy of 17 criteria organized into four categories, Spatial, Ergonomic, Activity, and Environmental, grounded in interior-design literature [Kilmer and Kilmer 2024; Panero 1962], that conforms to the personas and their activities. This revisits the classical idea of codifying design rules for layout generation [Leimer et al. 2022; Merrell et al. 2011; Yu et al. 2011], but makes the selected rules customized to the specific functional specifications, a customization that is only possible thanks to the foundational knowledge of LLMs. Finally, we generate an initial layout and iteratively refine it through a check-and-repair loop to make the layout conform to the parsed design guidelines.

Our functionality-aware pipeline generates diverse scenes from real-world functional specifications, with results preferred over all baselines and ablations in 94.3% of pairwise comparisons in a crowdsourced two-alternative forced choice (2AFC) perceptual study.

In summary, our contributions are:

- • A functionality-first framing for 3D indoor layout generation, shifting the input from object-centric prompts to functional specifications, and exposing failure modes not addressed by existing scene synthesis methods.
- • A design constraint taxonomy (Spatial, Ergonomic, Activity, Environmental) rooted in interior-design literature, together

with an LLM-driven method for automatically customizing constraints to specific occupant personas and activities. • An iterative layout generation framework that combines geometric measurements, LLM-based reasoning, and VLMbased visual assessment in a tool-augmented check-andrepair loop to generate high-quality, functionally valid layouts from functional specifications.

2 Related Works

Scene synthesis pre-LLMs. Indoor scene synthesis has long been studied in computer graphics. Early works leaned heavily on prespecified design principles [Merrell et al. 2011], simple statistical relationships [Yu et al. 2011] and hand-written programs [Yeh et al. 2012], which all require heavy manual effort, and is not comprehensive enough to handle open-ended settings, even when massively scaled up [Deitke et al. 2022; Raistrick et al. 2024]. As a result, the field gradually shifted towards data-driven approaches [Fisher et al. 2012; Kermani et al. 2016; Liang et al. 2017], and become dominated by deep learning based approaches [Bai et al. 2025; Li et al. 2019; Lin and Mu 2024; Paschalidou et al. 2021; Ritchie et al. 2019; Tang et al.

- 2024; Wang et al. 2019, 2018; Zhang et al. 2020; Zhou et al. 2019]. Such learning-based approaches require minimal manual effort, but generally do not learn explicit design principles, making them hard to adapt for our settings. LayoutEnhancer [Leimer et al. 2022] attempts to address this limitation by directly injecting ergonomic principles into learned generative models, yet they also involve manual rule authoring that is hard to scale. A parallel line of work has argued that scene plausibility should be grounded in human use. SceneGrok [Savva et al. 2014] predicted where actions occur from observations, activity-centric synthesis [Fisher et al. 2015] used such predictions to guide scene generation, and PiGraphs [Savva et al. 2016] learned joint models of human pose and scene geometry from interactions. Subsequent work conditioned layout on activity types [Fu et al. 2017; Ma et al. 2016; Qi et al. 2018] or optimized for human-aware navigation [Sun et al. 2023]. These methods established that activities can usefully guide object selection and placement, but they map activities directly to arrangements rather than to the underlying design constraints that determine whether a layout actually supports the activity.

LLM-based scene synthesis. While text-to-scene generation long predates [Chang et al. 2015; Ma et al. 2018] large language models, the rise of LLMs transformed the scale and flexibility of what language-conditioned systems can produce. LayoutGPT [Feng et al. 2023] showed that LLMs can directly predict object coordinates from open-vocabulary prompts, Holodeck [Yang et al. 2024c] scaled this to full embodied environments with constraint satisfaction, and I-Design [Çelen et al. 2024] added personalization from user preferences. Since then, the field has expanded rapidly: hierarchical and structured representations [Öcal et al. 2024; Pun et al. 2025; Tam et al. 2025; Wang et al. 2025; Wu et al. 2025; Zhang et al. 2025; Zhou et al. 2025], deeper spatial reasoning via chain-of-thought [Ran et al. 2025] or VLM-guided search [Berdoz et al. 2025; Deng et al. 2025], design-aware placement [Bucher and Armeni 2025; Feng et al.

- 2025; Gupta et al. 2026; Yang et al. 2025b], and agentic pipelines with iterative self-correction [He et al. 2026; Liu et al. 2025; Luo

[Figure 11]

- Fig. 2. Constraints Taxonomy. We organize interior design constraints into four categories: Spatial (S1–S5), Ergonomic (E1–E4), Activity (A1–A4), and Environmental (N1–N4), each illustrated with representative examples of how they shape furniture placement in a typical room layout.

et al. 2026; Sun et al. 2025b; Xia et al. 2026; Yang et al. 2025a]. Yet throughout this progression, the input prompt is predominantly about objects, relation and coordinates. Even systems that incorporate richer signals—personalized preferences [Çelen et al. 2024; Yang et al. 2024b], physical interactability [Yang et al. 2024a], or disentangled semantic-physical refinement [Gao et al. 2025; Pan and Liu 2025]—optimize primarily for spatial plausibility and visual coherence rather than for the functional, ergonomic, and environmental criteria that determine whether a room actually supports the activities it was designed for.

Scene evaluation and optimization. How a generated layout is evaluated shapes what it can become. Current evaluation frameworks assess geometric validity, semantic coherence, navigability, and collision avoidance [Hwangbo et al. 2025; Tam et al. 2026], and iterative optimization loops—VLM-based feedback [Asano et al. 2025; Jiang et al. 2026], multi-turn RL [Zhao et al. 2026], differentiable VLM optimization [Sun et al. 2025a], and VL-guided editing [Bian et al. 2025]—have made it possible to progressively refine layouts after initial generation [Feng et al. 2026]. Recent work has begun extending evaluation toward functional affordance grounding [Maillard et al. 2026], but existing frameworks still lack systematic coverage of the human-centered criteria—ergonomic fit, activity support, environmental comfort—that govern how people actually use indoor spaces. Our framework addresses this by employing typed verification tools—geometric checks for measurable spatial properties, LLM queries for contextual semantic judgments, and VLM assessments for holistic visual quality—organized across four constraint categories to evaluate and iteratively refine layouts against functional design criteria.

3 Design Constraints

Professional interior designers compose layouts by following a rich set of principles and guidelines drawn from both design literature and practical experience [Kilmer and Kilmer 2024; Panero 1962]. Prior work has formalized such guidelines as optimization criteria for automated layout generation [Merrell et al. 2011], yet these criteria are applied uniformly regardless of who occupies the space or what they do in it.

In practice, layout requirements are not universal. A mobilitylimited user and a frequent entertainer sharing the same room

type will impose fundamentally different demands on furniture placement, clearance, and zone allocation. We capture this variability through activity × persona combinations derived from the user’s functional specifications, which together determine which constraints are relevant and how their thresholds are parameterized for a given scene. We detail this process further in Section 4. We ground our constraints taxonomy in established interior design literature and organize constraints into four categories: Spatial, Ergonomic, Activity, and Environmental. We illustrate these constraints in Figure 2 and Table 1:

Spatial. Forming the foundation upon which all other constraint categories depend, these constraints govern the placement and relational arrangement of furniture within the room. Geometry Validity (S1) requires that every furniture piece fit within the room boundary without overlapping adjacent objects, except where a containment or nesting relationship is explicitly defined between them; Boundary & Attachment (S2) specifies that floor-based items rest on the floor plane, wall-mounted objects attach to the correct surface, and large case goods align to the nearest wall; Spatial Relationships (S3) captures the grouping logic central to interior design practice, where functionally paired objects must be placed in proximity and with appropriate relative orientation; Scale & Proportion (S4) ensures furniture size is commensurate with room dimensions and neighboring objects, preventing pieces from dominating or failing to define the space; Visual Composition (S5) captures aesthetic principles such as focal point orientation, visual balance, and alignment, ensuring the arrangement reads as intentional rather than arbitrary.

Ergonomic. Wherespatialconstraintsestablish what fits, ergonomic

constraints ensure the space can be safely and comfortably navigated and used, with thresholds parameterized to the physical needs and abilities of the target persona. Circulation (E1) requires that primary pathways maintain a minimum clear width for unobstructed passage, with thresholds elevated for users relying on mobility aids; Interaction Clearance (E2) ensures that all articulated elements have full action zones free of obstruction, and that seating has adequate pull-out space behind it; Reachability (E3) constrains frequently used objects and controls to fall within the user’s operational height range, accounting for seated versus standing use and any physical limitations; Body Fit & Posture (E4) requires work

Table 1. Evaluation tools setup. Each constraint is verified by one or more tools, color-coded by type: numeric/geometric tools compute quantitative measures directly from scene geometry, LLM query tools leverage language model reasoning over structured scene data, and VLM tools interpret rendered images. Tier indicates evaluation priority, where lower-tier constraints are verified first as prerequisites for higher-tier ones.

Category Constraints Tools What it checks (Return) Tier

- S1 Geometry Validity

boundary_check() within-wall containment (bool);

T1 bbox_collison() pairwise overlap ratio (%)

- S2 Boundary & Attachment

contact_check() floor/ceiling/wall attachment (bool);

T1 wall_angle_check() object-to-wall angle (degrees)

- S3 Spatial Relationships

object_exist() object presence (bool);

T2 object_info() object size, location and orientation data (𝑙, 𝑤,ℎ, 𝑥, 𝑦,𝑧, facing)

- S4 Scale & Proportion

size_ratio() object-to-room size ratio (%);

T2 size_check() LLM judgement on absolute size plausibility

- S5 Visual Composition visual_balance_check() VLM judgement on top-down visual balance assessment T5

Spatial

- E1 Circulation

pathfinding() the path (list of (𝑥,𝑧) world-coordinate waypoints) or null;

T2 path_width() minimum clearance (m) and bottleneck position (𝑥,𝑧)

- E2 Interaction Clearance

articulation_zone() minimum clearance within swing arc (m);

T4 chair_clearance() front and rear clearance distances from chair (m)

- E3 Reachability reach_check() LLM judgement on reachability given user’s attributes and object_info() T4

- E4 Body Fit & Posture posture_check() LLM judgement on posture given user’s attributes and object’s size T4

Ergonomic

- A1 Activity Zone

free_floor_area() free area within the zone (m2) ;

T3

object_in_zone() LLM judgment on if related objects are correctly positioned within the zone; activity_support_check() LLM judgment on if related objects can support the required activity well

- A2 Sightlines & Privacy inbetween_check() LLM judgement on if objects blocks between two points based on object_info() T3

- A3 Workflow Sequencing

total_path_length() total path length across activity sequence (m);

T5 workflow_check() LLM judgment on workflow order

- A4 Multi-activity Compat multi_activity_check() LLM judgement on if space supports multi-activity simultaneously reusing tools for A1 T5

Activity

- N1 Natural Light Access window_obs_ratio() object within proximity radius blocking window ratio (%) T6

- N2 Glare Prevention

screen_window_info() angle and distance between screen and window;

T5 glare_check() LLM judgment on glare risk

- N3 Acoustic Separation

zone_distance() distance between two zones (m);

T5 acoustic_check() LLM judgment on acoustic risk

- N4 Ventilation & Thermal

Environmental

vent_obs_ratio() object within proximity radius blocking vent ratio;

T6 distance_check() safe distance guarantee (bool)

surface heights, seat dimensions, and monitor distances to conform to anthropometric standards for sustained comfortable use.

Activity. Beyond physical validity and ergonomic fit, a layout must support the specific tasks the user performs in the space, which is the primary vehicle through which the activity × persona combination shapes the constraint pool. Zone Allocation (A1) ensures each primary activity has a dedicated zone of sufficient size, equipped with the relevant furniture and kept clearable and accessible when needed; Multi-Activity Compatibility (A2)

demands the layout support zone transformation without full reorganization when a space serves more than one purpose; Sightlines and Privacy (A3) encodes directional requirements, where certain tasks demand unobstructed views toward an entry or a child’s play area, while others call for visual shielding from the rest of the room; Workflow Sequencing (A4) arranges objects to match their order of use, avoiding backtracking and cross-circulation.

Environmental. Furniture placement shapes not only how a room is used but how it feels, through its effect on light, sound, and thermal comfort. Natural Light Access (N1) asks that furniture

[Figure 12]

- Fig. 3. Overall Pipeline: Given a functional prompt, Function2Scene generates 3D indoor scene layout through iteratively evaluation and refinement based on functional constraints.

arrangement preserve daylight reach into primary activity zones; Glare Prevention (N2) guards against screens and resting surfaces receiving direct window light during typical use hours; Acoustic Separation (N3) requires for noise-generating activities to be spatially buffered from quiet zones through distance, furniture mass, or deliberate zone boundaries; Ventilation & Thermal Comfort (N4) ensures furniture does not block windows, and that temperaturesensitive activities are positioned away from cold drafts or direct heat sources.

Constraints are assigned to one of six priority tiers (T1 to T6) as indicated in Table 1, where lower tiers must be satisfied before higher tiers are considered. Details of how each constraint is evaluated are described in Section 4.2.

- 4 Method

As illustrate in Figure 3, the pipeline proceed through two main stages: (1) the Initialization stage, which parses the user’s functional prompt into a parsed scene description and a list of functional constraints, constructs the room structure with user verification, and generates an initial furniture layout; and (2) the Constraints-based Evaluation and Refinement stage, which iteratively evaluates the layout against constraints in priority order using specialized tools, and applies targeted adjustments guided by design principles. Following a similar prompting strategy in Holodeck [Yang et al. 2024c], an LLM prompt is designed for each stage throughout the pipeline, consisting of a task description, an ouput format specification, and a one-shot example.

In this section, we first introduce the initialization stage, which parses the user’s functional prompt into an initial layout and a set of functional constraints. We then provide a detailed description of the constraints-based layout evaluation and refinement stage, which iteratively optimizes the layout toward functional and spatial coherence.

Our system is a function-driven interior design agent that takes a functional prompt as input, which captures the user’s living needs as input distilled through multi-turn conversations, and produces a furniture layout tailored to how they actually inhabit the space, guided by design principles derived from those needs.

- 4.1 Initialization

The initialization stage serves as the foundation of Function2Scene, transforming a user-provided functional description of a scene into a structured 3D layout that serves as the starting point for the evaluation and refinement steps. It consists of three steps: parsing, room structure generation, and furniture initialization.

Parsing. Given the raw functional description of a scene, the parser extracts: (a) a structured set of constraints grounded in the 4 category constraint taxonomy defined in Section 3; and (b) a parsed scene description that serves as an LLM-friendly reformulation of the original input, analyzed and restructured with functional constraints considerations in mind to produce a more rational and unambiguous description in a style closer to that of current text-toscene generation models. Together, these two outputs ensure that the original functional intent is interpreted once and propagated consistently throughout the pipeline.

Room Structure Generation. Given the parsed description, Function2Scene initializes the room structure through a set of LLM prompts. The room structure is encoded in a custom JSON-based Domain-Specific Language (DSL), as shown in the cyan box in Figure 3, which provides a well-defined representation of walls, floors, ceilings, doors and windows, with explicit geometric attributes such as dimension and coordinates, and semantic attributes such as orientation encoded as a facing direction. This structural output is visualized for user to verify, reflecting the natural client-designer workflow where the room structure is reviewed and approved before furnishing begins. If corrections are needed, the user can easily refine the geometry through natural language prompting, making the editing process intuitive and accessible.

Furniture Initialization. With the verified empty room and the parsed description, Function2Scene directly generates an initial layout.However,aswedemonstrate in our evaluation, LLM-generated layouts at this stage are fundamentally limited in their spatial reasoning: objects may overlap, violate functional adjacency requirements, or produce configurations that are physically plausible but practically unusable, as shown in the initialization stage of Figure 3. The furniture initialization therefore serves as a starting point rather than a final result, motivating the constraints-based evaluation and refinement stage that follows.

- 4.2 Constraints-based Evaluation and Refinement

Prior works have explored constraint-based layout generation, most notably Holodeck [Yang et al. 2024c], which defines a set of spatial relational constraints (e.g. in front of, near, face to) and optimizes object placements to satisfy them. While effective for basic spatial arrangement, such approaches are restricted to geometric relations between objects, leaving ergonomic, activity, and environmental demands unaddressed.

Function2Scene extends this idea to a richer, 4-category constraint set derived directly from the user’s functional input, as detailed in Section 3. Rather than optimizing all constraints simultaneously through a solver, we adopt an LLM-driven iterative evaluation loop that assesses constraints sequentially in priority order, as specified in Table 1.

Constraint Evaluation. For each constraint, specialized tools are invoked to retrieve structured spatial information, which the LLM interprets to determine whether the constraint is satisfied. Specifically, the agent first interprets the constraint description in the context of the current layout state, then selects and invokes the appropriate tool(s) such as pathfinding(), visual_balance_check(), or posture_check() to retrieve varied forms of feedback, ranging from numeric measurements and traversal paths to natural language assessments from numeric algorithm, VLM or LLM-based tools. The LLM then interprets these results against the constraint requirements and produces a justification along with a proposed refinement step if the constraint is not met.

Layout Refinement. For each unsatisfied constraint, the LLM generates a targeted refinement action based on the justification produced in the evaluation step. Refinement guidance is grounded in a set of design principles [Panero 1962] covering both universal spatial standards and room-specific human factor recommendations. Universal principles include maintaining a minimum 36” primary circulation path and preserving door swing clearances. Room-specific standards further constrain the refinement: in the bedroom, a minimum 2’0”–3’0” side clearance around the bed is required for access and bed-making; in the dining room, at least 3’4” must be preserved behind seated diners for service circulation; and in the living room, conversation groups should be arranged within 8 feet of each other for comfortable interaction. These standards inform how furniture should be repositioned, reoriented, or resized. Each adjustment is applied locally to avoid disrupting already-satisfied constraints, and the affected constraint is re-evaluated before the agent proceeds to the next one.

Termination. Constraints are evaluated sequentially in priority order. If resolving a constraints would introduce violations in higherpriority constraints that have already been satisfied, the constraint is skipped. Once all constraints have been assessed, the Tier 1 spatial constraints are re-evaluated to verify that no adjustments made in later stages have compromised foundational layout quality. The final layout is then returned as output.

5 Results and Evaluation

In thissection, wepresentqualitative and quantitative results demonstrating the ability of our method in generating visually and functionally plausible indoor scenes given functionality-focused natural language prompts. We compare against three representative LLM-based layout generation methods: Holodeck [Yang et al. 2024c], iDesign [Çelen et al. 2024], and LayoutVLM [Sun et al. 2025a], which cover the current landscape of language-driven indoor scene synthesis from persona-aware generation to open-ended spatial instruction following. Our evaluation focuses on how well each approach captures persona-specific functional requirements and lifestyle-driven spatial organization that standard benchmarks often overlook.

- 5.1 Data

We curated 30 real interior design cases from Architectural Digest [Architectural Digest 2026], an internationally recognized magazine and authority on interior design. Each case describes a distinct room designed around a specific occupant persona, resulting in a diverse dataset spanning 10 room types, including bedrooms, kitchens, living rooms, dining rooms, studios/ateliers, a home library, a guestroom, a nursery, a great room, and a mezzanine; 30 unique personas ranging from a retired couple and a chef to a drag queen, a child with autism, and a YouTuber. This breadth ensures coverage of varied functional needs, aesthetic preferences, lifestyle contexts, and demographic profiles.

- 5.2 Perceptual Study

To evaluate how well different layout generation methods satisfy the functional requirements specified in user scene prompts, we conducted a two-alternative forced-choice (2AFC) perceptual study. We recruited 30 participants through Prolific [Prolific 2026], with diverse backgrounds prior experience with AI evaluation tasks. Each participant completed 30 scene comparisons, where each comparison presented a room brief alongside rendered images of two layouts in randomized order, and selected the more functional layout for the described occupants. We also randomly insert 5 attention checks that compares against randomly generated, implausible layouts, and filter out participants who fail any of these checks.

We have a total of 10 comparison conditions (6 baselines, 4 ablations), resulting in 3 answers per scene and condition. For each comparison, we aggregate responses across all valid participants who saw that pair and report the proportion of selections in favour of our method. To reduce evaluation time and focus judgments on layout quality rather than scene-level details, participants were instructed to prioritize structural validity, such as furniture blocking doorways or objects extending outside room boundaries before considering brief-specific criteria.

Table 2 shows the results of this experiment, and Figure 4 shows some qualitative comparisons between generated layouts. Overall, participants preferred layouts generated using our method across all baselines and prompt conditions, with an aggregate preference rate of 94.3%. Against Holodeck, our method was preferred in 92.2% and 88.9% of trials under functional and parsed prompts respectively. Against iDesign, preference rates reached 94.4% and 98.9%, with the parsed condition yielding the highest score across all comparisons. Against LayoutVLM, our method was preferred in 96.7% and 94.4% of trials. Figure 1 and Figure 5 provides a closer look at our generated layouts, highlighting fine-grained constraint satisfaction across spatial and functional requirements. Please refer to the supplementary material for more detailed visualization of our results and comparisons.

Furthermore, Table 3 presents ablation study results examining the contribution of each component in our pipeline. Notably, retaining iterative updates without evaluation tools performs worst than removing both, indicating that iterative refinement is counterproductive without grounded spatial feedback to guide it. Additionally, prompt format has negligible effect when tools are absent, confirming that richer constraint representations only enforces their benefit

- Table 2. 2AFC study results comparison our method with baselines. Each baseline is ran with both the original functional prompt and our parsed scene description and constraints.

Method Prompt % Ours preferred Holodeck [Yang et al. 2024c]

Functional 92.2 Parsed 88.9

iDesign [Çelen et al. 2024]

Functional 94.4 Parsed 98.9

LayoutVLM [Sun et al. 2025a]

Functional 96.7 Parsed 94.4

Overall — 94.3

- Table 3. 2AFC study results comparing against ablations that uses different input and generation strategy.

Prompt format Iterative update Evaluation Tools %Ours preferred

Functional No No 83.3 Parsed No No 83.3

Functional Yes No 78.9 Parsed Yes No 80.0 Parsed Yes Yes Ours

when paired with the tool set that measures them. These results establish evaluation tools as the critical enabler of our pipeline.

6 Conclusion

In this paper, we presented Function2Scene, a framework for generating indoor layouts from functional specifications. By focusing on functionality, we take a first step towards designing a LLM-driven scene generation pipeline that better suits real interior design workflows, demonstrating that a well-designed taxonomy of functional design principles, combined with a LLM-driven iterative pipeline, can produce higher quality, more functional scenes than those generated by prior works.

There exist many opportunities in further extending our method: as stated in the introduction, our method starts from a professionally written, detailed functional specification. Real design workflows, however, usually begin with user demands that are much vaguer and shorter: users need designers’ help to discover, articulate, and refine their needs through multiple rounds of conversation and feedback. A conversational interface that helps non-expert users arrive at a detailed functional specification would complete the upstream half of the design workflow that feeds into our method; while our constraint taxonomy provides a general framework for evaluating functionality, our current verification protocol heavily relies on basic numeric checks and LLM queries. They could be made much more powerful with more constraint-specific tools, for example, embodied simulation with articulated models, physically accurate lighting and acoustic estimations, or a domain-specific language for expressing distance and dimension requirements in more semantic manners; more broadly, our framework currently operates within a fixed architectural shell in residential environments, co-optimizing room shape, openings, and partitions alongside furniture placement would better capture the full scope of interior design.

References

Architectural Digest. 2026. Room Tours & Interior Design Case Studies. https://www. architecturaldigest.com. Accessed: 2026.

Yuto Asano, Naruya Kondo, Tatsuki Fushimi, and Yoichi Ochiai. 2025. From geometry to culture: An iterative vlm layout framework for placing objects in complex 3d scene contexts. arXiv preprint arXiv:2503.23707 (2025).

Tongyuan Bai, Wangyuanfan Bai, Dong Chen, Tieru Wu, Manyi Li, and Rui Ma. 2025. FreeScene: Mixed Graph Diffusion for 3D Scene Synthesis from Free Prompts. In Proceedings of the Computer Vision and Pattern Recognition Conference. 5893–5903.

Frédéric Berdoz, Luca A Lanzendörfer, Nick Tuninga, and Roger Wattenhofer. 2025. Text-to-Scene with Large Reasoning Models. arXiv preprint arXiv:2509.26091 (2025).

Zixuan Bian, Ruohan Ren, Yue Yang, and Chris Callison-Burch. 2025. HOLODECK 2.0: Vision-Language-Guided 3D World Generation with Editing. arXiv preprint arXiv:2508.05899 (2025).

Martin JJ Bucher and Iro Armeni. 2025. ReSpace: Text-Driven 3D Scene Synthesis and Editing with Preference Alignment. arXiv preprint arXiv:2506.02459 (2025).

Ata Çelen, Guo Han, Konrad Schindler, Luc Van Gool, Iro Armeni, Anton Obukhov, and Xi Wang. 2024. I-design: Personalized llm interior designer. In European Conference on Computer Vision. Springer, 217–234.

Angel Chang, Will Monroe, Manolis Savva, Christopher Potts, and Christopher D Manning. 2015. Text to 3d scene generation with rich lexical grounding. In Proceedings of the 53rd Annual Meeting of the Association for Computational Linguistics and the 7th International Joint Conference on Natural Language Processing (Volume 1: Long Papers). 53–62.

Matt Deitke, Eli VanderBilt, Alvaro Herrasti, Luca Weihs, Kiana Ehsani, Jordi Salvador, Winson Han, Eric Kolve, Aniruddha Kembhavi, and Roozbeh Mottaghi. 2022. ProcTHOR: Large-Scale Embodied AI Using Procedural Generation. Advances in Neural Information Processing Systems 35 (2022), 5982–5994.

Wei Deng, Mengshi Qi, and Huadong Ma. 2025. Global-local tree search in vlms for 3d indoor scene generation. In Proceedings of the Computer Vision and Pattern Recognition Conference. 8975–8984.

Haoran Feng, Yifan Niu, Zehuan Huang, Yang-Tian Sun, Chunchao Guo, Yuxin Peng, and Lu Sheng. 2026. Repurposing 3D Generative Model for Autoregressive Layout Generation. arXiv preprint arXiv:2604.16299 (2026).

Weitao Feng, Hang Zhou, Jing Liao, Li Cheng, and Wenbo Zhou. 2025. CasaGPT: cuboid arrangement and scene assembly for interior design. In Proceedings of the Computer Vision and Pattern Recognition Conference. 29173–29182.

Weixi Feng, Wanrong Zhu, Tsu-jui Fu, Varun Jampani, Arjun Akula, Xuehai He, Sugato Basu, Xin Eric Wang, and William Yang Wang. 2023. Layoutgpt: Compositional visual planning and generation with large language models. Advances in Neural Information Processing Systems 36 (2023), 18225–18250.

Matthew Fisher, Daniel Ritchie, Manolis Savva, Thomas Funkhouser, and Pat Hanrahan.

2012. Example-based synthesis of 3D object arrangements. ACM Transactions on Graphics (TOG) 31, 6 (2012), 1–11.

Matthew Fisher, Manolis Savva, Yangyan Li, Pat Hanrahan, and Matthias Nießner.

2015. Activity-centric scene synthesis for functional 3D scene modeling. ACM Transactions on Graphics (TOG) 34, 6 (2015), 1–13.

Qiang Fu, Xiaowu Chen, Xiaotian Wang, Sijia Wen, Bin Zhou, and Hongbo Fu. 2017. Adaptive synthesis of indoor scenes via activity-associated object relation graphs. ACM Transactions on Graphics (TOG) 36, 6 (2017), 1–13.

Jialin Gao, Donghao Zhou, Mingjian Liang, Lihao Liu, Chi-Wing Fu, Xiaowei Hu, and Pheng-Ann Heng. 2025. DisCo-Layout: Disentangling and Coordinating Semantic and Physical Refinement in a Multi-Agent Framework for 3D Indoor Layout Synthesis. arXiv preprint arXiv:2510.02178 (2025).

Kunal Gupta, Ishit Mehta, Kun Wang, Nicholas Chua, Abhimanyu Krishna, Yan Deng, Ravi Ramamoorthi, and Manmohan Chandraker. 2026. INTERIORAGENT: LLM Agent for Interior Design aware 3D Layout Generation. In Thirteenth International Conference on 3D Vision.

Yun He, Kelin Yu, and Matthias Zwicker. 2026. SceneOrchestra: Efficient Agentic 3D Scene Synthesis via Full Tool-Call Trajectory Generation. arXiv preprint arXiv:2604.19907 (2026).

Gyeom Hwangbo, Hyungjoo Chae, Minseok Kang, Hyeonjong Ju, Soohyun Oh, and Jinyoung Yeo. 2025. LEGO-Eval: Towards Fine-Grained Evaluation on Synthesizing 3D Embodied Environments with Tool Augmentation. arXiv preprint arXiv:2511.03001 (2025).

Haiyan Jiang, Deyu Zhang, Dongdong Weng, Weitao Song, and Henry Been-Lirn Duh.

2026. HOG-Layout: Hierarchical 3D Scene Generation, Optimization and Editing via Vision-Language Models. arXiv preprint arXiv:2604.10772 (2026).

Z Sadeghipour Kermani, Zicheng Liao, Ping Tan, and Hao Zhang. 2016. Learning 3D scene synthesis from annotated RGB-D images. In Computer Graphics Forum, Vol. 35. Wiley Online Library, 197–206.

Rosemary Kilmer and W Otie Kilmer. 2024. Designing interiors. John Wiley & Sons. Kurt Leimer, Paul Guerrero, Tomer Weiss, and Przemyslaw Musialski. 2022. Layouten-

hancer: Generating good indoor layouts from imperfect data. In SIGGRAPH Asia 2022 Conference Papers. 1–8.

Manyi Li, Akshay Gadi Patil, Kai Xu, Siddhartha Chaudhuri, Owais Khan, Ariel Shamir, Changhe Tu, Baoquan Chen, Daniel Cohen-Or, and Hao Zhang. 2019. Grains: Generative recursive autoencoders for indoor scenes. ACM Transactions on Graphics (TOG) 38, 2 (2019), 1–16.

Yuan Liang, Song-Hai Zhang, and Ralph Robert Martin. 2017. Automatic data-driven room design generation. In International Workshop on Next Generation Computer Animation Techniques. Springer, 133–148.

Chenguo Lin and Yadong Mu. 2024. Instructscene: Instruction-driven 3d indoor scene synthesis with semantic graph prior. arXiv preprint arXiv:2402.04717 (2024). Xinhang Liu, Yu-Wing Tai, and Chi-Keung Tang. 2025. Agentic 3D Scene Generation with Spatially Contextualized VLMs. arXiv preprint arXiv:2505.20129 (2025).

Jun Luo, Jiaxiang Tang, Ruijie Lu, and Gang Zeng. 2026. SceneAssistant: A Visual Feedback Agent for Open-Vocabulary 3D Scene Generation. arXiv preprint arXiv:2603.12238 (2026).

Rui Ma, Honghua Li, Changqing Zou, Zicheng Liao, Xin Tong, and Hao Zhang. 2016. Action-driven 3D indoor scene evolution. ACM Trans. Graph. 35, 6 (2016), 173–1.

Rui Ma, Akshay Gadi Patil, Matthew Fisher, Manyi Li, Sören Pirk, Binh-Son Hua, Sai-Kit Yeung, Xin Tong, Leonidas Guibas, and Hao Zhang. 2018. Language-driven synthesis of 3D scenes from scene databases. ACM Transactions on Graphics (TOG) 37, 6 (2018), 1–16.

Léopold Maillard, Francis Engelmann, Tom Durand, Boxiao Pan, Yang You, Or Litany, Leonidas Guibas, and Maks Ovsjanikov. 2026. SceneTeract: Agentic Functional Affordances and VLM Grounding in 3D Scenes. arXiv preprint arXiv:2603.29798 (2026).

Paul Merrell, Eric Schkufza, Zeyang Li, Maneesh Agrawala, and Vladlen Koltun. 2011. Interactive furniture layout using interior design guidelines. ACM transactions on graphics (TOG) 30, 4 (2011), 1–10.

Başak Melis Öcal, Maxim Tatarchenko, Sezer Karaoğlu, and Theo Gevers. 2024. Sceneteller: Language-to-3d scene generation. In European Conference on Computer Vision. Springer, 362–378.

Zhenyu Pan and Han Liu. 2025. MetaSpatial: Reinforcing 3D Spatial Reasoning in VLMs

for the Metaverse. arXiv preprint arXiv:2503.18470 (2025). Julius Panero. 1962. Anatomy for interior designers. Whithey Library of Design. Despoina Paschalidou, Amlan Kar, Maria Shugrina, Karsten Kreis, Andreas Geiger, and

Sanja Fidler. 2021. Atiss: Autoregressive transformers for indoor scene synthesis. Advances in neural information processing systems 34 (2021), 12013–12026.

Prolific. 2026. Easily collect high quality data from real people. https://www.prolific. com/. Accessed: 2026.

Hou In Derek Pun, Hou In Ivan Tam, Austin T Wang, Xiaoliang Huo, Angel X Chang, and Manolis Savva. 2025. HSM: Hierarchical Scene Motifs for Multi-Scale Indoor Scene Generation. arXiv preprint arXiv:2503.16848 (2025).

Siyuan Qi, Yixin Zhu, Siyuan Huang, Chenfanfu Jiang, and Song-Chun Zhu. 2018. Human-centric indoor scene synthesis using stochastic grammar. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 5899–5908.

Alexander Raistrick, Lingjie Mei, Karhan Kayan, David Yan, Yiming Zuo, Beining Han, Hongyu Wen, Meenal Parakh, Stamatis Alexandropoulos, Lahav Lipson, et al. 2024. Infinigen indoors: Photorealistic indoor scenes using procedural generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 21783–21794.

Xingjian Ran, Yixuan Li, Linning Xu, Mulin Yu, and Bo Dai. 2025. Direct Numerical Layout Generation for 3D Indoor Scene Synthesis via Spatial Reasoning. arXiv preprint arXiv:2506.05341 (2025).

Daniel Ritchie, Kai Wang, and Yu-an Lin. 2019. Fast and flexible indoor scene synthesis via deep convolutional generative models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 6182–6190.

Manolis Savva, Angel X Chang, Pat Hanrahan, Matthew Fisher, and Matthias Nießner.

2014. SceneGrok: Inferring action maps in 3D environments. ACM transactions on graphics (TOG) 33, 6 (2014), 1–10.

Manolis Savva, Angel X Chang, Pat Hanrahan, Matthew Fisher, and Matthias Nießner.

2016. Pigraphs: learning interaction snapshots from observations. ACM Transactions On Graphics (TOG) 35, 4 (2016), 1–12.

Fan-Yun Sun, Weiyu Liu, Siyi Gu, Dylan Lim, Goutam Bhat, Federico Tombari, Manling Li, Nick Haber, and Jiajun Wu. 2025a. Layoutvlm: Differentiable optimization of 3d layout via vision-language models. In Proceedings of the Computer Vision and Pattern Recognition Conference. 29469–29478.

Fan-Yun Sun, Shengguang Wu, Christian Jacobsen, Thomas Yim, Haoming Zou, Alex Zook, Shangru Li, Yu-Hsin Chou, Ethem Can, Xunlei Wu, et al. 2025b. 3D-Generalist: Self-Improving Vision-Language-Action Models for Crafting 3D Worlds. arXiv preprint arXiv:2507.06484 (2025).

Jia-Mu Sun, Jie Yang, Kaichun Mo, Yu-Kun Lai, Leonidas Guibas, and Lin Gao. 2023. Haisor: Human-Aware Indoor Scene Optimization via Deep Reinforcement Learning. ACM Trans. Graph. (2023). Just Accepted.

Hou In Ivan Tam, Hou In Derek Pun, Austin T. Wang, Angel X. Chang, and Manolis Savva. 2025. SceneMotifCoder: Example-driven Visual Program Learning for Generating 3D Object Arrangements. In Proceedings of the IEEE Conference on 3D Vision (3DV).

Hou In Ivan Tam, Hou In Derek Pun, Austin T Wang, Angel X Chang, and Manolis Savva. 2026. SceneEval: Evaluating semantic coherence in text-conditioned 3D indoor scene synthesis. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision. 7355–7365.

Jiapeng Tang, Yinyu Nie, Lev Markhasin, Angela Dai, Justus Thies, and Matthias Nießner. 2024. Diffuscene: Denoising diffusion models for generative indoor scene synthesis. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 20507–20518.

Kai Wang, Yu-An Lin, Ben Weissmann, Manolis Savva, Angel X Chang, and Daniel Ritchie. 2019. Planit: Planning and instantiating indoor scenes with relation graph and spatial prior networks. ACM Transactions on Graphics (TOG) 38, 4 (2019), 1–15.

Kai Wang, Manolis Savva, Angel X Chang, and Daniel Ritchie. 2018. Deep convolutional priors for indoor scene synthesis. ACM Transactions on Graphics (TOG) 37, 4 (2018), 1–14.

Xiping Wang, Yuxi Wang, Mengqi Zhou, Junsong Fan, and Zhaoxiang Zhang. 2025. HLG: Comprehensive 3D Room Construction via Hierarchical Layout Generation. arXiv preprint arXiv:2508.17832 (2025).

Qirui Wu, Denys Iliash, Daniel Ritchie, Manolis Savva, and Angel X. Chang. 2025. Diorama: Unleashing Zero-shot Single-view 3D Indoor Scene Modeling. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV). 8896–8907. Hongchi Xia, Xuan Li, Zhaoshuo Li, Qianli Ma, Jiashu Xu, Ming-Yu Liu, Yin Cui, Tsung-Yi Lin, Wei-Chiu Ma, Shenlong Wang, Shuran Song, and Fangyin Wei. 2026. SAGE: Scalable Agentic 3D Scene Generation for Embodied AI. arXiv preprint arXiv:2602.10116 (2026).

Yandan Yang, Baoxiong Jia, Shujie Zhang, and Siyuan Huang. 2025a. Sceneweaver: All-in-one 3d scene synthesis with an extensible and self-reflective agent. arXiv preprint arXiv:2509.20414 (2025).

Yandan Yang, Baoxiong Jia, Peiyuan Zhi, and Siyuan Huang. 2024a. PhyScene: Physically Interactable 3D Scene Synthesis for Embodied AI. In Proceedings of Conference on Computer Vision and Pattern Recognition (CVPR).

Yixuan Yang, Junru Lu, Zixiang Zhao, Zhen Luo, James JQ Yu, Victor Sanchez, and Feng Zheng. 2024b. Llplace: The 3d indoor scene layout generation and editing via large language model. arXiv preprint arXiv:2406.03866 (2024).

Yixuan Yang, Zhen Luo, Tongsheng Ding, Junru Lu, Mingqi Gao, Jinyu Yang, Victor Sanchez, and Feng Zheng. 2025b. OptiScene: LLM-driven Indoor Scene Layout Generation via Scaled Human-aligned Data Synthesis and Multi-Stage Preference Optimization. In The Thirty-ninth Annual Conference on Neural Information Processing Systems.

Yue Yang, Fan-Yun Sun, Luca Weihs, Eli VanderBilt, Alvaro Herrasti, Winson Han, Jiajun Wu, Nick Haber, Ranjay Krishna, Lingjie Liu, et al. 2024c. Holodeck: Language guided generation of 3d embodied ai environments. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 16227–16237.

Yi-Ting Yeh, Lingfeng Yang, Matthew Watson, Noah D Goodman, and Pat Hanrahan.

2012. Synthesizing open worlds with constraints using locally annealed reversible jump mcmc. ACM Transactions on Graphics (TOG) 31, 4 (2012), 1–11.

Lap-Fai Yu, Sai Kit Yeung, Chi-Keung Tang, Demetri Terzopoulos, Tony F Chan, and Stanley J Osher. 2011. Make it home: Automatic optimization of furniture arrangement. ACM Trans. Graph. 30, 4 (2011), 86.

Guangyao Zhai, Evin Pınar Örnek, Shun-Cheng Wu, Yan Di, Federico Tombari, Nassir Navab, and Benjamin Busam. 2023. Commonscenes: Generating commonsense 3d indoor scenes with scene graph diffusion. Advances in Neural Information Processing Systems 36 (2023), 30026–30038.

Yunzhi Zhang, Zizhang Li, Matt Zhou, Shangzhe Wu, and Jiajun Wu. 2025. The scene language: Representing scenes with programs, words, and embeddings. In Proceedings of the Computer Vision and Pattern Recognition Conference. 24625–24634.

Zaiwei Zhang, Zhenpei Yang, Chongyang Ma, Linjie Luo, Alexander Huth, Etienne Vouga, and Qixing Huang. 2020. Deep generative modeling for scene synthesis via hybrid representations. ACM Transactions on Graphics (TOG) 39, 2 (2020), 1–21. Yang Zhao, Shizhao Sun, Meisheng Zhang, Yingdong Shi, Xubo Yang, and Jiang Bian.

2026. SceneReVis: A Self-Reflective Vision-Grounded Framework for 3D Indoor Scene Synthesis via Multi-turn RL. arXiv preprint arXiv:2602.09432 (2026).

Mengqi Zhou, Xipeng Wang, Yuxi Wang, and Zhaoxiang Zhang. 2025. RoomCraft: Controllable and Complete 3D Indoor Scene Generation. arXiv preprint arXiv:2506.22291

(2025).

Yang Zhou, Zachary While, and Evangelos Kalogerakis. 2019. Scenegraphnet: Neural message passing for 3d indoor scene augmentation. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 7384–7392.

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

Kids friendly kitchen for antiques businessman, 3.5m × 5m, L-shaped. Kid does homework at the island. Display collection of vintage kichenware. Meetup with clients and dealers

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

8 year old boy's bedroom, 3m x 3.5m, build models and reading with friends. blackout sleeping, storage and models display

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

Collector's home library, 4m x 5m, storage for large collections, family gathering with fireplace, board games for 6 people

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

Freelance creative couples dining room, 3.5m x 4m, ceramics painting/drying, reading, photo editing, gear storage

[Figure 38]

[Figure 39]

[Figure 40]

Ours Holodeck I-Design LayoutVLM

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

Remote worker's studio, 5.3m x 5.3m, daily video calls, split working and eating, window views, friends gathering and staying the night.

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

Drag queen's bedroom, 3m x 5m, large performance wardrobe, geing ready spot, flea market collector, outfit checks

Direct parsed w con. & iter.

Direct functional

Ours

Direct parsed

- Fig. 4. Qualitative comparisons of our method against various comparison conditions. Top two rows: baselines with original functional prompts; middle two rows: baselines with our parsed specifications; bottom two rows: ablations, from left to right: w/ parsed input and iterative refinement, with original prompt and no iterative refinement, with parsed input and no iterative refinement.

[Figure 55]

|[Figure 56]<br><br>[Figure 57]|
|---|

|[Figure 58]<br><br>[Figure 59]|
|---|

Spare room (4m x 5m) used as a home oﬀice on weekdays and a movie room on weekend evenings with 2-4 friends. Daily work needs a proper desk with dual monitors and good task lighting. Friends occasionally stay the night aer party

|[Figure 60]<br><br>[Figure 61]|
|---|

|[Figure 62]<br><br>[Figure 63]|
|---|

[Figure 64]

|[Figure 65]<br><br>[Figure 66]|
|---|

|[Figure 67]<br><br>[Figure 68]|
|---|

Antique buyer's guest room (3m x 3.5m). As a monthly traveler, I need to unload luggage in this room. A proper sofa to lie flat would be good for evening TV and eating alone. Would like to display antique pieces from buying trips and need a table for examining pieces under morning sunlight. Friends stay on portable air maress occasionally

|[Figure 69]<br><br>[Figure 70]|
|---|

|[Figure 71]<br><br>[Figure 72]|
|---|

[Figure 73]

|[Figure 74]<br><br>[Figure 75]|
|---|

|[Figure 76]<br><br>[Figure 77]|
|---|

Writer's nursery (3m x 3.5m). First baby girl and need easy access to the crib. Street noise and strong light control is necessary. Occasionally long night feeds needs a glider to rest and close enough to reach the crib in dark. Storage space for baby stuﬀs. So lighting only

|[Figure 78]<br><br>[Figure 79]|
|---|

|[Figure 80]<br><br>[Figure 81]|
|---|

- Fig. 5. Functional scenes generated by our method, along with zoomed in highlights. The input prompts are truncated due to space constraints. Please refer to the supplementary materials for all qualitative results, along with visualization of all intermediary optimization steps.

We provide a GUI showing the completed iterations and results in index_static.html in the supplementary. We also provide a GUI showing all qualitative comparisons in qualitative.html.

- A Layout Representation

To formally specify room layouts, we define a Domain-Specific Language (DSL) encoded in JSON. This DSL provides a structured, machine-readable representation of a room’s architectural surfaces and furnishings, designed to be both human-authored and amenable to automated layout validation and 3D rendering.

The DSL represents a room as a JSON object with two top-level arrays: room_structure and furniture. The room_structure array enumerates all architectural surfaces: walls, floor, ceiling, door, and window, while furniture enumerates all movable objects placed within the room. A short _convention block at the top of each file defines the shared coordinate system and field semantics that apply throughout.

All geometry uses a right-handed coordinate system with axes +𝑋 (east), −𝑋 (west), +𝑌 (up), and +𝑍 (south). Every surface and object carries a location field giving its centroid in world space, and a dimensions field expressed as [width, height, depth] in metres in the element’s local frame. Orientation is encoded by a facing field—an integer bearing in [0◦, 360◦) representing the direction the element’s outward or inward normal points, following compass convention (0◦ = north, 90◦ = east, 180◦ = south, 270◦ = west). Walls use their inward normal as the facing value; floor and ceiling omit the field entirely, as their normals are fixed by convention. Openings such as doors and windows are represented either as standalone entries with physical dimensions, or as holes sub-arrays on the parent wall, giving a 2D offset and size within that wall’s local surface frame.

Each furniture entry additionally carries an orientation tag: "directional" for objects with a meaningful front face, "axial" for objects symmetric about one axis (e.g. a rug), and "symmetric" for fully rotationally symmetric objects such as a ceiling light. This distinction allows downstream rendering and constraint-solving code to apply appropriate symmetry assumptions when validating placement rules. The top-level description field is present only in refined or final scenes, where it records the set of layout constraints that have been resolved in the current iteration. It serves as a humanreadable invariant summarising the design intent that the scene satisfies: for example, asserting that the footprint of every personal storage unit is large enough to accommodate one occupant’s books, devices, and weekend gear without overflow.

- B Perceptual Study Details

StudyDesign. We conductedatwo-alternativeforced-choice (2AFC)

perceptual study to evaluate layout quality across 30 room scenes. Each scene was associated with one output from our method and outputs from 10 baseline methods or ablations (10 comparison types in total). To ensure each participant evaluated a diverse cross-section of scenes, each survey instance contained exactly 30 pairs, with each pair drawn from a different scene. Across all 30 pairs, the comparison method for each pair was sampled from the pool of 10 comparison types, ensuring broad coverage.

Fig. 6. Sample layout DSL file.

|{<br><br>"description": "S3 RULE: ...", "room_structure": [<br><br>{<br><br>"name": "wall_n", "location": [0, 1.2, -2.26], "dimensions": [3.5, 2.4, 0.02], "facing": 180, "color": "#F2F0EB", "holes": [{"location": [1.1, -0.15], "<br><br>dimensions": [0.9, 2.1]}] },<br><br>...<br><br>], "furniture": [<br><br>{<br><br>"name": "bunk_bed", "orientation": "directional", "location": [0.75, 0.825, -1.75], "dimensions": [2.0, 1.65, 1.0], "facing": 180, "color": "#C49464"<br><br>},<br><br>... ]<br><br>}<br><br>|
|---|

Attention Checks. 5 of the 30 pairs in each survey were attention check items, in which our method’s output was compared against a randomly generated layout. These checks were distributed among the 30 pairs and served to identify inattentive respondents. Any participant who failed to correctly identify the clearly superior layout on any attention check question was excluded from analysis.

Survey Variants. We fixed 10 distinct survey variants, each containing a different assignment of comparison methods across the 30 scenes. Each variant was completed by 3 participants, yielding a target of 30 submissions. Each survey was presented as a web-based interface (screenshots provided in Figure 7) in which participants viewed a brief describing the room’s intended occupants and use cases alongside two rendered scene images, and selected the better layout.

Recruitment. Participants were recruited via Prolific under an AI task evaluation study category. Eligibility criteria required: normal or corrected-to-normal vision, no colour blindness, current country of residence in the United States or Canada, at least 100 prior Prolific submissions, and at least 10 prior AI evaluation task submissions. These criteria were chosen to ensure participants had prior experience with structured evaluation tasks and could reliably perceive visual differences in the rendered scenes.

Participants. We collected 45 submissions in total, of which 32 passed all attention checks and were retained for analysis (valid response rate: 71.1%). The 32 valid participants varied in age, racial

[Figure 82]

Fig. 7. Perceptual study interface.

background, and occupation, providing a demographically diverse sample. We remove 2 repeated results.

Interface. The study interface presented each pair of rendered scenes side by side, accompanied by the room brief and persona description. Participants selected their preferred layout by clicking the image or a radio button below it, then advanced to the next pair. Four evaluation dimensions: spatial validity, ergonomics, activity support, and environmental quality were described in the onboarding instructions and available as hover-over reminder chips during the study as shown in Figure 8. Participants were instructed to prioritize structural issues (missing walls, doors, or windows; furniture blocking egress; objects extending outside room boundaries; lack of clear pathways) before consulting brief-specific criteria. The estimated completion time was 20–30 minutes (approximately 30 seconds per pair).

C Implementation Details C.1 Initialization

The room initialization pipeline converts a natural-language room description into a structured JSON scene through a sequence of discrete, LLM-driven steps. Each step is implemented as a structured prompt that instructs the language model to perform a narrowly scoped transformation, and intermediate outputs are saved to disk so that a human reviewer can inspect and correct the structure before the next stage begins.

Room shell generation. The pipeline opens by classifying the room as either rectangular or L-shaped based on linguistic cues in the description. Phrases indicating a structural recess: such as "alcove," "nook," "missing corner," or explicit references to two rectangular zones meeting at right angles, trigger the L-shaped path; all other descriptions default to a rectangle. Once the shape is determined, the shell prompt generates the room’s structural envelope: four walls for a rectangle, or six walls tracing the L’s perimeter for the L-shaped variant. Both formats share a common coordinate system (origin at the floor center, +X east, +Z south, +Y up) and produce a

[Figure 83]

Fig. 8. Perceptual study introduction.

consistent JSON schema with named wall entries, a floor polygon or slab, and a ceiling. Critically, every holes array is left empty at this stage, and no openings of any kind are cut into the walls.

Door and window placement. With the shell confirmed, the second step populates openings. Doors and windows are appended as new entries in the room_structure array, each carrying worldcoordinate position, dimensions, facing direction, and color. Simultaneously, a matching hole record is written into the parent wall’s holes array using the wall’s local coordinate frame, accounting for the fact that the local horizontal axis maps to different world directions depending on the wall’s facing. The step enforces a set of placement constraints: openings must fit within the wall’s extents, must not overlap one another, and windows must respect a mandatory 0.2 m header clearance below the wall top. Standard size presets are provided for both doors (single 0.9 m, double 1.5 m) and windows (standard through panoramic), with the system defaulting to larger windows to avoid a cramped feel.

Human verification. The pipeline deliberately pauses after door and window placement. The shell and the populated structure are written as separate files so that a reviewer can compare them and make manual edits: adding interior partitions, repositioning openings, or adjusting dimensions - before saving the final structural baseline. This checkpoint prevents furniture placement from proceeding against a structurally incorrect room.

A final prompt instructs the model to read the human-reviewed structural file and generate a furniture array describing every object in the scene. Each object entry is produced by the model with a center position, bounding-box dimensions, facing direction, and an orientation class (directional, axial, or symmetric) that signals how the object’s facing should be interpreted by downstream renderers. For multi-functional spaces such as studio apartments, the prompt provides the model with zone checklists: sleeping, living, kitchen, bathroom, work, and dining, ensuring that every implied

zone is populated with its essential objects even when the description names a zone without enumerating individual items. As with all prior steps, the model is instructed to leave the structural elements entirely untouched; furniture placement is strictly additive.

C.2 Functional Constraints

We define the parser to generate parsed scene description and a list of functional constraints based on the functional prompt. Here is an example of the complete constraints, more examples can be referred in the website we provided.

ZONE MAP ======== Zone | HARD or SOFT |

Boundary defined by Desk work area | SOFT | Wherever the desk , office chair , and task lamp cluster lands Movie viewing and lounge | SOFT | Wherever the screen , sofa , occasional seating , and coffee table cluster lands

Guest sleeping area | SOFT | Wherever the sofa bed or fold-out bed is deployed; overlaps the lounge cluster when a sofa bed is used

CONSTRAINTS ===========

TIER 1 - HARD

--------------

GROUP: Outer Boundary and Interior Walls Priority: S1 always first

-------------------------------------

S1 RULE: no floor-placed object extends beyond the outer room boundary APPLY TO: every floor-placed object in the

scene JSON PASS IF: overlap_ratio = 0 for every object IF FAIL: translate the failing object inward

along the axis of overlap until overlap_ratio = 0; recheck S1 overlap after move

EXCEPTIONS: NONE

S1 RULE: no wall-mounted object extends beyond the outer room boundary APPLY TO: every wall-mounted object in the

scene JSON PASS IF: overlap_ratio = 0 for every object IF FAIL: translate the failing object inward

along the wall until overlap_ratio = 0; verify the object still lies flush against

the wall after move EXCEPTIONS: NONE

- S1 RULE: no two floor-placed objects overlap each other

APPLY TO: every unordered pair of floor-placed

objects in the scene JSON PASS IF: overlap_ratio = 0 for every pair IF FAIL: translate the lower-priority object

in the pair away from the higher-priority object along the axis of greatest overlap until overlap_ratio = 0; recheck boundary rule after move

EXCEPTIONS: any chair or stool and its directly paired table or desk surface may have a non-zero overlap ratio consistent with the seat tucked fully under the table

apron or desk knee space; the executor determines which chairs are paired with which tables from the scene JSON cluster assignments; the ratio for every nonpaired object pair must be 0

GROUP: Doors , Fixtures , and Windows

- Priority: S2 > E1

-------------------------------------

--- FLOOR CONTACT AND WALL ATTACHMENT ---

- S2 RULE: every floor-based object rests flat on the floor plane APPLY TO: every floor-placed object in the

scene JSON PASS IF: Z = 0 for every object IF FAIL: snap the failing object to Z = 0 EXCEPTIONS: NONE

- S2 RULE: every wall-mounted object is flush against its backing wall APPLY TO: every wall-mounted object in the

scene JSON PASS IF: angle to nearest wall <= 5 degrees for every object IF FAIL: rotate and translate the failing object until it lies flat against the wall surface EXCEPTIONS: NONE

--- WALL-MOUNT VALIDITY ---

- S2 RULE: every wall-mounted object sits entirely on solid wall , not spanning a window opening , door opening , or corner void APPLY TO: every wall-mounted object in the

scene JSON; wall_segment is the specific wall section behind the object , with window and door openings masked as nonsolid

PASS IF: the object 's full horizontal and vertical span falls within solid wall only ; overlap with any opening = 0

IF FAIL: slide the object laterally along the wall until its full span clears every opening; if no solid span is long enough , flag for replacement with a narrower unit or relocation to an adjacent wall

EXCEPTIONS: curtain rails are exempt - they are by design wider than the window and span the opening; no other wall-mounted object is exempt

--- HINGED DOORS ---

S2 RULE: the swing arc of every hinged door is clear of all objects APPLY TO: every door in the scene that has a

physical door leaf; if door type is not stated , treat as hinged by default

PASS IF: swing arc polygon intersects no object bounding box

IF FAIL: move the blocking object out of the swing arc zone; if the object is fixed , flag the door as requiring a sliding conversion

EXCEPTIONS: NONE

S2 RULE: the approach path to every hinged or sliding door is unobstructed on both sides APPLY TO: every hinged or sliding door in the

scene JSON; entry_point is one body-width directly in front of the door on each side

PASS IF: a clear straight path exists from entry_point to door face on both sides

IF FAIL: move the blocking object away from the door face to restore the clear approach on the blocked side

EXCEPTIONS: NONE

--- OPEN THRESHOLDS ---

S2 RULE: no object occupies the clear-path zone of an open threshold between two zones APPLY TO: every opening between zones that has

no physical door leaf (open-plan archways , zone transitions described as open passages , wide openings between sub-areas ,

etc.)

PASS IF: a continuous clear path exists through the full width of the opening; no object centroid falls within the opening span; clear width along the path is wide enough for comfortable passage by a standard adult

IF FAIL: move the encroaching object laterally out of the threshold span; if the clear width is too narrow , move flanking objects

outward on both sides until width is sufficient

EXCEPTIONS: a floor lamp or small planter placed at the very edge of the threshold is acceptable if it does not reduce the clear span below comfortable passage width

for a standard adult

--- WINDOW OBSTRUCTION ---

- S2 RULE: no object blocks a window opening APPLY TO: every non-curtain , non-curtain-rail

object in the scene JSON; window_opening is the full rectangular face of each window in the room

PASS IF: overlap_ratio = 0 between every object and every window opening

IF FAIL: move the blocking object away from the window; if the object is a low unit designed to sit below a sill , run a VLM check to confirm it does not reach the glazing - if it does , replace with a shorter unit or move it to another wall

EXCEPTIONS: curtain rails and curtains are exempt; a radiator or built-in unit below the sill line may be reviewed by VLM to confirm it does not obstruct glazing flag for manual check if present

--- FIXTURE ORIENTATION ---

- S2 RULE: every fixture with a defined use-face is oriented so the use-face is accessible from

open floor APPLY TO: every fixture with a defined use-

face: sofa (seat faces the screen wall), occasional seating (seat faces the screen or the sofa front), desk (work surface faces a wall or window so the user has back support from open floor), television or projection screen (screen face is visible from the sofa and occasional seating)

PASS IF: angle between use-face normal and nearest unobstructed open-floor centroid <= 45 degrees

IF FAIL: rotate the fixture until its use-face points toward open floor; recheck S1 overlap and S2 wall-attachment after rotation EXCEPTIONS: NONE

TIER 2 - HIGH: Layout Plausibility

------------------------------------

GROUP: Circulation Priority: E1 > S3

-------------------------------------

- E1 RULE: every main circulation path through the room is passable for this user

APPLY TO: every pair of occupied zones between

which a resident must travel in daily use ; select pairs by identifying which zones are used in sequence for each activity in the scene (entry door to desk; entry door to sofa and screen; sofa to screen; sofa to entry door for guest movement on movie nights)

PASS IF: a clear path exists between every such pair; clear width at every point along the path is wide enough for a standard able-bodied adult and is wide enough to allow two to four guests to enter and reach the sofa cluster without single-file shuffling

IF FAIL: move the object(s) narrowing the path

outward from the path centerline until clear width is sufficient along the full length

EXCEPTIONS: NONE

E1 RULE: the entry zone immediately inside the main door has a clear sightline into the room APPLY TO: the main entry door; eye position is one step inside the door at standing height for a standard adult

PASS IF: no obstruction on the ray from entry eye to room interior centroid IF FAIL: move the obstructing object out of

the entry sightline cone EXCEPTIONS: NONE

E1 RULE: both long sides of any deployed sleeping surface are reachable for the guest who sleeps there APPLY TO: every sleeping surface in the scene

JSON (sofa bed in its unfolded state and any separate fold-out bed); check each long side independently in the deployed configuration

PASS IF: clear space along at least one long side is wide enough for a standard adult to sit on the edge and stand up; the foot end is also reachable on foot for bedding changes

IF FAIL: move the sleeping surface or adjacent

furniture in the deployed configuration to open the blocked side

EXCEPTIONS: a sofa bed used by a single guest

may have its back long side against a wall ; the front long side must remain clear in

the deployed configuration

E1 RULE: dead zones behind doors and in corners do not exceed an acceptable size for a standard adult APPLY TO: every corner and behind-door area in

the room; identify dead zones as areas bounded by walls and furniture with no access path

PASS IF: dead zone area is small enough to be non-functional storage at most

IF FAIL: move the furniture creating the dead zone outward from the corner to reduce dead zone size; or assign the dead zone as

intentional low-access storage and flag it in the scene

EXCEPTIONS: NONE

GROUP: Desk Work Cluster

- Priority: S3 > A3

-------------------------------------

- S3 RULE: objects that belong to the desk work cluster are placed in proximity to each other APPLY TO: the desk work cluster consisting of

the desk , the two monitors that sit on the

desk , the office chair , the task lamp , and any nearby office storage

PASS IF: every object in the cluster is within expected functional proximity of its cluster partners; no object from a different cluster intrudes between cluster

members

IF FAIL: move the outlying object toward its cluster centroid until the cluster is spatially coherent

EXCEPTIONS: NONE

- S3 RULE: objects within the desk work cluster share a common axis or facing alignment APPLY TO: every pair within the desk work

cluster that has a defined facing relationship (office chair faces desk; monitors face the user position at the desk; desk back is parallel to its supporting wall)

PASS IF: relative orientation delta between each pair is within an acceptable facing range (parallel or perpendicular as appropriate for the object type)

IF FAIL: rotate the misaligned object until alignment is confirmed; recheck S1 overlap

after rotation EXCEPTIONS: NONE

- S3 RULE: residual space created when the desk cluster is positioned is assessed for usability APPLY TO: every gap or leftover area created

between the desk cluster and the walls or adjacent clusters

PASS IF: residual area is either large enough for a meaningful function (circulation path , office storage , additional seat) or small enough to be negligible; no awkward mid-room residual strip exists

IF FAIL: shift the cluster to reduce awkward residual strips; or assign the residual area to a named function and flag it for potential use

EXCEPTIONS: NONE

S3 RULE: the footprint or span of every office storage unit is large enough to hold the office supplies designated for the desk zone without overflow APPLY TO: every office storage object near the

desk cluster (drawer unit , shelf , filing cabinet , etc.)

PASS IF: storage unit capacity is sufficient for the designated load based on footprint and volume ratio

IF FAIL: replace the storage unit with a larger unit , or add an additional unit of the same type in adjacent wall space

EXCEPTIONS: NONE

S3 RULE: adjacent wall space near the desk cluster is evaluated for expansion capacity if primary office storage proves insufficient APPLY TO: the desk cluster; residual_wall_span is the wall length adjacent to the cluster not yet occupied by furniture PASS IF: residual wall span is noted and its capacity for an additional shelf , cabinet , or rack unit is flagged

IF FAIL: flag the desk storage zone as expansion-constrained; recommend alternative storage locations in the scene

EXCEPTIONS: NONE

GROUP: Movie Viewing and Lounge Cluster Priority: S3 > A3

-------------------------------------

S3 RULE: objects that belong to the movie viewing and lounge cluster are placed in proximity to each other

APPLY TO: the movie viewing and lounge cluster

consisting of the television or projection screen , the sofa or sofa bed , any occasional seating (armchairs , floor cushions), the coffee table or low side surface , and the rug under the cluster PASS IF: every object in the cluster is within expected functional proximity of its cluster partners; no object from a different cluster intrudes between cluster

members

IF FAIL: move the outlying object toward its cluster centroid until the cluster is spatially coherent

EXCEPTIONS: NONE

- S3 RULE: objects within the movie viewing and lounge cluster share a common axis or facing alignment APPLY TO: every pair within the lounge cluster

that has a defined facing relationship ( sofa faces screen; occasional seating faces screen or sofa front; coffee table aligns with sofa front; rug aligns with sofa-screen axis)

PASS IF: relative orientation delta between each pair is within an acceptable facing range (parallel or perpendicular as appropriate for the object type)

IF FAIL: rotate the misaligned object until alignment is confirmed; recheck S1 overlap

after rotation EXCEPTIONS: NONE

- S3 RULE: residual space created when the lounge cluster is positioned is assessed for usability APPLY TO: every gap or leftover area created

between the lounge cluster and the walls or the desk cluster

PASS IF: residual area is either large enough for a meaningful function (circulation between desk and sofa , additional occasional seating , storage) or small enough to be negligible; no awkward midroom residual strip exists

IF FAIL: shift the cluster to reduce awkward residual strips; or assign the residual area to a named function and flag it for potential use

EXCEPTIONS: NONE

S3 RULE: the footprint or span of every storage unit near the lounge cluster is large enough to hold the items designated for the lounge and guest sleep zone without overflow APPLY TO: every storage object placed near the

lounge cluster intended to hold bedding for overnight guests , throws , remotes , and movie-night supplies PASS IF: storage unit capacity is sufficient for the designated load based on footprint and volume ratio

IF FAIL: replace the storage unit with a larger unit , or add an additional unit of the same type in adjacent wall space

EXCEPTIONS: NONE

S3 RULE: adjacent wall space near the lounge cluster is evaluated for expansion capacity if

primary lounge and bedding storage proves insufficient APPLY TO: the lounge cluster;

residual_wall_span is the wall length adjacent to the cluster not yet occupied by furniture or by the screen wall

PASS IF: residual wall span is noted and its capacity for an additional shelf , cabinet , or sideboard is flagged

IF FAIL: flag the lounge storage zone as expansion-constrained; recommend alternative storage locations in the scene

EXCEPTIONS: NONE

STANDALONE

-------------------------------------

S4 RULE: every piece of furniture has a footprint proportional to the room area and to the other objects around it APPLY TO: every floor-placed object; compare

each object 's footprint against the 20 square meter room area and against the footprint of its nearest neighbours

PASS IF: no single object dominates the room disproportionately; no object is so small relative to its neighbours that it reads as decorative rather than functional

IF FAIL: flag the disproportionate object for replacement with an appropriately scaled alternative; do not resize in place

EXCEPTIONS: NONE

S4 RULE: the length of the sofa or primary seating unit is proportional to the wall or zone it faces APPLY TO: the sofa or sofa bed in the lounge

cluster; measure its length against the width of the wall or zone face it is oriented toward (the screen wall) PASS IF: sofa length is neither so long it crowds the flanking space nor so short it leaves the facing wall visually empty; sofa length must also be sufficient to seat at least three of the two to four expected guests with the host occupying the fourth seat or an occasional chair IF FAIL: flag the sofa for replacement with an

appropriately sized unit; or reposition the sofa to face a wall of more proportionate width

EXCEPTIONS: NONE

TIER 3 - HIGH: Activity Support

---------------------------------

A1 - Desk work with dual monitors and task lighting

-------------------------------------

A1 RULE: the desk work zone has enough clear

floor area for the activity to be performed comfortably by a standard adult seated at a desk

APPLY TO: the desk work zone; the activity pose is a seated adult at the desk with the office chair pulled to working position and reaching for both monitors and the task lamp; the activity pose is checked against the computed free-floor polygon

PASS IF: the free-floor polygon within the zone contains the seated and chair-pulledout body pose without collision

IF FAIL: move furniture at the perimeter of the zone outward to expand the free-floor polygon until the activity pose fits within the free-floor polygon

EXCEPTIONS: NONE

A1 RULE: the desk work zone is accessible from the main circulation path APPLY TO: the entry point of the desk work

zone

PASS IF: a clear path exists from the main circulation path to the desk zone entry point; path is wide enough for a standard adult

IF FAIL: move the object blocking the desk zone entry point to restore the access path

EXCEPTIONS: NONE

A1 RULE: the desk work zone is permanently clear and is not encroached on by movie viewing or sleeping furniture in any room mode APPLY TO: every object whose footprint or

articulation might cross the desk work zone in lounge mode or in deployedsleeping mode

PASS IF: in every room mode the desk , office chair pullout zone , and task lamp remain unobstructed; no sofa bed or fold-out bed deployment footprint , no guest seating , and no occasional chair encroaches on the desk work zone

IF FAIL: relocate the encroaching object to a position that does not cross the desk work

zone in any mode; if no such position exists , flag the room as overloaded and recommend reducing furniture count or replacing pieces with smaller equivalents

EXCEPTIONS: NONE

A1 - Movie watching with two to four friends

-------------------------------------

A1 RULE: the movie viewing zone has enough clear floor area for two to four guests to be seated facing the screen comfortably

APPLY TO: the movie viewing and lounge zone; the activity pose is two to four seated adults distributed across the sofa and occasional seating with feet on the floor or on a coffee table; the activity pose is

checked against the computed free-floor polygon

PASS IF: the free-floor polygon within the zone contains all seated body poses without collision and provides foot space in front of each seat

IF FAIL: move furniture at the perimeter of the zone outward to expand the free-floor polygon until all seated activity poses fit within the free-floor polygon; if no expansion is possible , reduce the occasional seating count to the maximum that fits

EXCEPTIONS: NONE

A1 RULE: the movie viewing zone is accessible from the main circulation path APPLY TO: the entry point of the movie viewing

zone

PASS IF: a clear path exists from the main circulation path to the front of the sofa and to each occasional seat; path is wide enough for a standard adult and allows multiple guests to enter and seat themselves without queueing through a single bottleneck

IF FAIL: move the object blocking the zone entry or any individual seat approach to restore the access path

EXCEPTIONS: NONE

A1 RULE: if the movie viewing zone is also used as the overnight sleeping zone , the furniture that must be moved or unfolded for the transition can be repositioned without violating S1 or S2 in either configuration APPLY TO: every object involved in the lounge-

to-sleep transition (sofa bed unfold motion , coffee table relocation , occasional chair stowage)

PASS IF: each object has a valid resting position in both the lounge configuration and the deployed sleeping configuration that satisfies S1 and S2

IF FAIL: identify an alternative resting position for the object that satisfies S1 and S2 in both configurations; if none exists , replace the sofa with a sofa bed of a smaller deployed footprint , or move the coffee table to a permanent stowed position along a free wall

EXCEPTIONS: NONE

A1 - Overnight sleeping for guests

-------------------------------------

A1 RULE: the overnight sleeping zone has enough clear floor area for one adult sleeping surface in deployed configuration APPLY TO: the overnight sleeping zone in its

deployed configuration; the activity pose is a standard adult lying full-length on the sleeping surface with at least one long side reachable on foot; the activity pose is checked against the computed freefloor polygon in the deployed configuration

PASS IF: the deployed sleeping surface fits inside the room without violating S1 boundary or pairwise overlap; the freefloor polygon along at least one long side

accommodates a standing adult

IF FAIL: move furniture at the perimeter of the sleeping zone outward in the deployed configuration to expand the free-floor polygon; if the room cannot accommodate the deployed footprint , replace the sleeping surface with a narrower or shorter unit

EXCEPTIONS: NONE

A1 RULE: the overnight sleeping zone is accessible from the main circulation path and from the entry door APPLY TO: the entry point of the deployed

sleeping zone

PASS IF: a clear path exists from the main circulation path to the side of the deployed sleeping surface; path is wide enough for a standard adult

IF FAIL: move the object blocking the sleep

zone entry to restore the access path EXCEPTIONS: NONE

A1 RULE: the lounge-mode furniture that must be moved or unfolded to create the overnight sleeping zone can be repositioned without violating S1 or S2 in the cleared configuration APPLY TO: every object that must be moved or

stowed to deploy the overnight sleeping zone (coffee table , occasional chairs , rug

edges , sofa cushions in sofa-bed mode) PASS IF: each object has a valid resting

position in the cleared configuration that satisfies S1 and S2

IF FAIL: identify an alternative resting position for the object that satisfies S1 and S2; if none exists , flag the sleeping zone as requiring a permanently smaller default lounge footprint

EXCEPTIONS: NONE

GROUP: Zone Transformation Priority: A1 > A4

-------------------------------------

A4 RULE: when the room transitions between office

mode , movie mode , and overnight sleeping mode , every object that must be repositioned can move to a valid new position without violating

S1, S2, or A1 in the transformed state

APPLY TO: every object involved in any mode transformation between the three activity modes (office chair pushed under desk for movie mode; coffee table relocated for sleep deployment; sofa bed unfolded for sleep mode; occasional chairs stowed for sleep mode; rug edge cleared for sofa bed deployment)

PASS IF: after repositioning , S1 and S2 rules pass for all objects in the scene in every mode; the activity pose fits in the transformed layout for the active mode IF FAIL: adjust the default position of the

object so that its transformation path does not cause S1 or S2 violations; or replace the object with a version that folds or stacks to a smaller footprint

EXCEPTIONS: NONE

A4 RULE: rugs or mats that must be rolled back or

cleared for the sofa bed deployment can be rolled without moving furniture that sits on their edges APPLY TO: every rug in the scene that is used

under furniture in lounge mode and must also be clearable to allow the sofa bed to

unfold

PASS IF: the rug can be rolled or cleared on the deployment side without requiring the repositioning of furniture that rests on its edges

IF FAIL: move the furniture off the rug edges in the default lounge layout so the rug is

independently rollable; recheck S1 overlap after adjustment

EXCEPTIONS: NONE

TIER 4 - HIGH: Ergonomic Fit

------------------------------

GROUP: Desk Work Access Priority: E2 > E1

-------------------------------------

E2 RULE: every door , drawer , and openable panel has a clear sweep zone in front of it APPLY TO: every hinged door , drawer , cabinet

door , and openable panel in the scene JSON , including any drawer unit or filing cabinet near the desk; the executor identifies these by object type from the scene description

PASS IF: articulation zone polygon intersects no other object 's bounding box

IF FAIL: move the object encroaching on the articulation zone away from the openable element until the zone is clear

EXCEPTIONS: NONE

- E2 RULE: the office chair at the desk has a clear pullout zone behind it

APPLY TO: every chair , stool , and seat in the scene JSON that is pushed in against a desk or table surface , including the office chair at the desk

PASS IF: chair pullout zone intersects no other object 's bounding box IF FAIL: move the object behind the chair away until the pullout zone is clear; if the room does not allow this , flag the seating

count as excessive for the available space

EXCEPTIONS: NONE

E2 RULE: no companion object is placed within another object 's access zone or articulation zone APPLY TO: every pair of objects in the scene

JSON where one has an access zone and the other is a companion placed nearby (task lamp base inside the chair pullout zone; office storage drawer face within the desk

chair pullout zone; coffee table within the sofa front access zone)

PASS IF: no companion object bounding box overlaps any other object 's access zone polygon

IF FAIL: move the companion object out of the

access zone; prefer moving it laterally along the wall rather than further into the room EXCEPTIONS: NONE

GROUP: Lounge and Guest Sleep Access Priority: E2 > E1

-------------------------------------

E2 RULE: every openable element near the lounge and guest sleep cluster has a clear sweep zone in front of it

APPLY TO: every hinged door , drawer , cabinet door , sofa bed unfold mechanism , and storage panel in or adjacent to the lounge

cluster

PASS IF: articulation zone polygon intersects no other object 's bounding box in both lounge and deployed-sleep configurations

IF FAIL: move the object encroaching on the articulation zone away from the openable element until the zone is clear in both configurations

EXCEPTIONS: NONE

- E2 RULE: every occasional seat and the sofa has a clear front access zone for sitting and

standing APPLY TO: every chair , stool , and seat in the

lounge cluster including the sofa and any occasional armchairs

PASS IF: the seat 's front access zone polygon intersects no other object 's bounding box

IF FAIL: move the object encroaching on the front access zone away from the seat; if the coffee table is the encroaching object , move it forward away from the sofa until

the zone is clear EXCEPTIONS: NONE

- E2 RULE: no companion object is placed within another object 's access zone or articulation zone in the lounge cluster APPLY TO: every pair of objects in the lounge

cluster where one has an access zone and the other is a companion placed nearby ( coffee table inside sofa front access zone ; floor cushion inside occasional chair pullout zone; rug edge inside sofa bed deployment zone)

PASS IF: no companion object bounding box overlaps any other object 's access zone polygon

IF FAIL: move the companion object out of the access zone; prefer moving it laterally rather than further into the room

EXCEPTIONS: NONE

STANDALONE

-------------------------------------

- E3 RULE: every storage object is reachable by a standard adult without a step stool APPLY TO: every storage object in the scene

JSON (office shelves , drawer units , filing

cabinet , lounge sideboard or shelf , bedding storage); persona is matched to the user profile , a standard able-bodied adult

PASS IF: a reach check confirms that the highest and lowest storage positions are within the reach envelope of a standard adult

IF FAIL: lower the shelf or cabinet to bring the top shelf within reach; or raise the bottom drawer above knee level; flag if structural constraints prevent adjustment

EXCEPTIONS: NONE

- E3 RULE: the number of storage units currently specified is sufficient for the stated storage

load; if load exceeds capacity , additional units must be added before layout is finalised

APPLY TO: every named storage cluster in the scene (office storage near the desk and bedding-plus-lounge storage near the sofa) ; the executor sums available storage volume across all units in each cluster and compares against the estimated load derived from daily office work supplies and from bedding for two to four overnight

guests on weekend evenings

PASS IF: total storage volume in each cluster meets or exceeds the estimated load for that cluster

IF FAIL: add one or more storage units of the same type to the cluster; recheck S3 storage adequacy and S2 wall-mount validity for each added unit

EXCEPTIONS: NONE

E4 RULE: every seat and work surface is dimensioned for a standard adult 's body and posture APPLY TO: every chair , stool , sofa , sofa bed ,

and desk in the scene JSON; persona matched to the user profile , a standard able-bodied adult

PASS IF: seat height is appropriate for a standard adult to sit and stand from comfortably; legroom under the desk is sufficient for a standard adult 's leg length

IF FAIL: flag the object for replacement with a correctly sized alternative; do not resize in place

EXCEPTIONS: NONE

E4 RULE: every screen or monitor is positioned at

a comfortable viewing angle for a standard adult at the seated or standing position used for that screen APPLY TO: every monitor , television , and

projection screen in the scene JSON; eye_pos at the desk is the seated office position for the dual monitors; eye_pos at

the lounge is the seated sofa or occasional-seat position for the movie screen

PASS IF: vertical and horizontal viewing angles are within a comfortable range for a standard adult at each eye_pos; no extreme neck flexion or rotation required between the two monitors at the desk or to

the movie screen at the sofa IF FAIL: adjust screen height or tilt; or move

the desk or sofa position to bring the viewing angle within range; recheck E2 access zones after move

EXCEPTIONS: NONE

- TIER 5 - MEDIUM

-----------------

STANDALONE

-------------------------------------

S5 RULE: the room reads as visually balanced when viewed from above APPLY TO: the full room layout; the executor renders the top-down view and passes it to

VLM for balance and focal-point assessment

PASS IF: VLM confirms no extreme visual weight imbalance between room quadrants; visual

balance score is within an acceptable range for a multi-function spare room

IF FAIL: move heavy visual elements (sofa , screen wall cluster , dense shelving) to counterbalance the dominant quadrant; recheck S1 after any move

EXCEPTIONS: open-plan rooms and rooms with

strong asymmetric architectural features ( bay window , alcove on one side) may have a

lower symmetry threshold - VLM assessment takes precedence over the balance score

A3 RULE: the workflow sequence for each multistep activity follows a logical spatial path with minimal backtracking APPLY TO: every activity from Step 1B that has

a defined sequence of object interactions (office mode: entry door to desk to

office chair to monitors and task lamp; movie mode: entry door to sofa to screen and back , with detour to coffee table for snacks; sleep deployment: clear coffee table , unfold sofa bed , retrieve bedding from storage); steps[] is the ordered list

of objects touched during each activity

PASS IF: no backtracking or cross-path collision is found between steps in any activity; total path length is reasonable for a 20 square meter room

IF FAIL: rearrange the objects in the sequence

to reduce backtracking; prefer a linear or U-shaped path order; recheck E1 circulation and S1 overlap after rearrangement

EXCEPTIONS: NONE

N2 RULE: no screen or sustained visual-work surface receives direct glare from a window or artificial light source APPLY TO: every monitor , television ,

projection screen , and reading position in the scene; eye_pos is the user 's position at each (seated at desk for the two

monitors; seated on sofa or occasional seat for the movie screen)

PASS IF: the angle check confirms no window is

in the direct forward arc of each screen normal; VLM glare check finds no visible light source in the point-of-view render from each eye_pos; the task lamp at the desk does not cast direct glare onto either monitor surface from the user 's viewpoint

IF FAIL: rotate the screen or desk to move the

window out of the forward arc; or add a curtain or blind to the window causing glare; reposition the task lamp so its beam does not cross either monitor face; recheck S2 wall-mount validity if any blind mount is added

EXCEPTIONS: NONE

N3 RULE: noise-generating zones are acoustically separated from quiet zones by distance or mass obstruction

APPLY TO: every pair of zones in this room where one is a noise source and the other is a quiet zone; the movie viewing cluster

is the noise source in movie mode , and the desk work zone is the quiet zone in office mode; the executor checks separation between these two clusters PASS IF: separation distance between the desk and the screen-sofa axis is sufficient that the desk is not directly in front of the movie speakers , or a mass obstruction (sofa back , shelving unit) sits on the direct path between the screen and the desk

IF FAIL: move a mass object (sofa back , bookshelf , storage cabinet) onto the direct path between the screen and the desk; or increase separation distance by repositioning either cluster

EXCEPTIONS: NONE

- TIER 6 - LOW

-------------

STANDALONE

-------------------------------------

N1 RULE: priority activity zones have reasonable proximity to natural light APPLY TO: every activity zone from Step 1B

that benefits from natural light; priority

order in this room: desk work > movie viewing and lounge > overnight sleeping

PASS IF: the desk work zone centroid is within

an acceptable distance of a window so that daily work benefits from daylight; VLM brightness check confirms adequate daylight at the desk during typical weekday work hours based on sun path analysis

IF FAIL: move the desk cluster toward the nearest window so daylight reaches the work surface; recheck E1 circulation and N2 glare after any move

EXCEPTIONS: the overnight sleeping zone does not require window proximity - it may be positioned away from windows intentionally for blackout purposes; the movie viewing zone may also be positioned away from windows to reduce screen glare

N4 RULE: no object 's bounding box overlaps the clearance zone of an HVAC vent , radiator , or heat source APPLY TO: every object near a known HVAC vent ,

radiator , or fixed heat source identified in the scene description; if no heat source is identified , flag the room as needing manual confirmation of vent and radiator locations before final placement

PASS IF: the clearance check confirms no bounding box overlaps the vent clearance zone; the proximity check confirms no sleep or sustained-work surface is within the exclusion radius of a heat source IF FAIL: move the encroaching object away from

the vent or heat source until clearance is restored

EXCEPTIONS: NONE

C.3 Evaluation Tools

We provide implementation details of tools used for evaluation.

def boundary_check(obj, room): """ Check if object's bounding box lies within the

outer wall polygon. Returns: bool """

def bbox_collision(obj_a, obj_b): """ Compute AABB pairwise overlap ratio between

two objects. Returns: float , overlap percentage (0 means no

collision) """

def contact_check(obj, surfaces): """ Check if object is in contact with its

required surface (floor/wall/ceiling). Returns: bool """

def wall_angle_check(obj, walls): """ Compute angle between object facing and

nearest wall normal. Returns: float , angle in degrees

"""

def object_exist(name, scene): """ Check if a named object is present in the

scene. Returns: bool """

def object_info(name, scene): """ Retrieve geometric state of a named object. Returns: dict {dimensions [l,w,h], location [x

,y,z], facing (degrees)} """

def size_ratio(obj, room): """ Compute object footprint as a percentage of

total room floor area. Returns: float , percentage """

def size_check(obj): """ LLM judgement on whether object's absolute

size is plausible for its type. Returns: str, one of {normal , big , small} """

def visual_balance_check(scene): """ VLM judgement on room's visual balance from a

top -down rendered view. Returns: str, free -text assessment of balance ,

focal point , and alignment """

def pathfinding(start, end, scene, resolution=0.05 ): """ Find shortest navigable path between two floor

positions using A*. Args: start , end -- (x, z) world coordinates in metres Returns: list of (x, z) waypoints , or None if

no path exists """

def path_width(path, scene): """ Compute minimum navigable clearance along a

given path. Returns: dict {min_width (m), bottleneck (x, z

)} """

def articulation_zone(obj, scene): """ Compute minimum clearance within the swing arc

of a door or drawer.

Returns: float , minimum clear distance in

metres (0 means blocked) """

def chair_clearance(chair, table, scene): """ Measure front and rear clearance distances for

a chair. Returns: dict {front_clearance (m),

rear_clearance (m)} """

def reach_check(persona, obj): """ LLM judgement on reachability given user

attributes and object_info(). Returns: str , one of {reachable , too_high ,

too_low , too_deep} """

def posture_check(persona, obj): """ LLM judgement on posture given user attributes

and object dimensions. Returns: str , one of {fits , too_tight ,

poor_posture} """

def free_floor_area(zone, scene): """ Compute unoccupied floor area within a zone by

subtracting object footprints. Returns: float , free area in m^2 """

def object_in_zone(zone, activity, scene): """ LLM judgement on whether activity -relevant

objects are correctly placed in zone. Returns: str , one of {correctly_placed ,

misplaced , missing} """

def activity_support_check(activity, objects): """ LLM judgement on whether objects can

adequately support the required activity. Returns: str , one of {supported , unsupported ,

undersized , oversized} """

def inbetween_check(point_a, point_b, scene): """ LLM judgement on sightline obstruction between

two points using object_info(). Returns: dict {status: clear/blocked ,

blocker_id: str or None} """

def total_path_length(activity_sequence, scene): """

Compute cumulative travel distance across an

ordered activity sequence. Returns: float , total path length in metres """

def workflow_check(activity_sequence, scene): """ LLM judgement on whether object arrangement

follows a logical workflow order. Returns: str, one of {optimal , suboptimal ,

backtracking , cross_path} """

def multi_activity_check(activities, scene): """ LLM judgement on whether space supports

multiple activities simultaneously. Reuses free_floor_area(), object_in_zone(),

activity_support_check() per activity. Returns: dict {status: compatible/conflict ,

conflicting_pair: list or []} """

def window_obs_ratio(window, scene, radius): """ Compute per -object blocking ratio for objects

within proximity of a window. Returns: list of dict {id: str , blocking_ratio

: float}, sorted descending """

def screen_window_info(screen, window): """ Compute geometric relationship between a

screen and a window. Returns: dict {angle (degrees), distance (m)} """

def glare_check(screen, window): """ LLM judgement on glare risk based on screen -

window angle and distance. Returns: str, one of {no_risk , glare_risk} """

def zone_distance(zone_a, zone_b): """ Compute Euclidean distance between two zone

centroids. Returns: float , distance in metres """

def acoustic_check(zone_a, zone_b, scene): """ LLM judgement on acoustic separation risk

between a noise and quiet zone. Reuses activity_support_check() to verify sound -blocking objects on path. Returns: str, one of {separated , acoustic_risk

} """

def vent_obs_ratio(vent, scene, radius): """ Compute per -object blocking ratio for objects

within proximity of a vent. Returns: list of dict {id: str , blocking_ratio

: float}, sorted descending """

def distance_check(obj, heat_source, min_distance= 0.5): """ Check that a sensitive object maintains safe

clearance from a heat source. Returns: bool """

