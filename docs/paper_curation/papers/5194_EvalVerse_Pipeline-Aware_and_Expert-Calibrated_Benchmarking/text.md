# arXiv:2605.23271v1[cs.CV]22May2026

## EvalVerse: Pipeline-Aware and Expert-Calibrated Benchmarking for Professional Cinematic Video Generation

Songlin Yang1,2,†,∗, Haobin Zhong2,†, Ruilin Zhang3,†, Xiaotong Zhao2, Shuai Li2, Kai Zheng2, Xuyi Yang1, Zhe Wang1,2, Zhenchen Tang2,4, Yang Li2,4, Bohai Gu1,2, Zhengwei Peng2, Yidan Huang5, Mengzhou Luo5, Yihang Bo5, Dalu Feng5, Yujia Zhang2, Juntao Ma2, Ruiqi Wang2, Lvmin Zhang6, Yuwei Guo7, Frank Guan8, Maneesh Agrawala6, Hongbo Fu1, Alan Zhao2, Anyi Rao1,‡ 1The Hong Kong University of Science and Technology, 2Tencent, 3Tsinghua University, 4Institute of Automation, Chinese Academy of Sciences, 5Beijing Film Academy, 6Stanford University, 7The Chinese University of Hong Kong, 8Singapore Institute of Technology syangds@connect.ust.hk, anyirao@ust.hk

### Abstract

The rapid evolution of generative video foundation models has propelled the field toward professional-grade cinematic synthesis. To achieve such demanding quality, the community transitions towards Reinforcement Learning (RL) and agentic workflows. However, reliable evaluation has emerged as a critical bottleneck. Existing benchmarks predominantly evaluate “whether it is right” (basic promptfollowing) while fundamentally neglecting “whether it is good” (cinematic quality, acting, and aesthetics). Furthermore, current automated metrics lack the domainspecific rigor required to provide trustworthy signals, creating a severe credibility gap between human aesthetic perception and machine scoring. To bridge this gap, we introduce EvalVerse, a comprehensive, pipeline-aware, and expert-calibrated evaluation framework. We treat video generation assessment not merely as an engineering task, but as a core scientific problem: the systematic digitization of subjective cinematic expertise. First, we organize domain knowledge into an evaluation taxonomy aligned with the professional filmmaking workflow (preproduction, production, and post-production). Second, we distill human expert judgments into a curated dataset with large-scale human annotations. Third, we inject this knowledge into Vision-Language Models (VLMs) through an expertcalibrated fine-tuning strategy, enabling the VLM to perform explicit Chain-ofThought reasoning. Compared to previous works, EvalVerse not only retains compatibility with foundational “rightness” metrics, but also significantly expands the criteria to “goodness” and broaden the task coverage to complex multi-shot sequencing and audio-visual integration. Consequently, by providing granular diagnostic signals, EvalVerse transcends a static leaderboard and establishes a fundamental infrastructure for future work, such as reward models and evaluator agent.

∗Project leader †Equal contribution ‡Corresponding author

Preprint.

Step-II: Dataset Curation

###### Step-I: Taxonomy Establishment

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

3 Stages 7 Cinematic Aspects 18 Main Dim. 45 Sub Dim.

Database

Annotation

Million-level high-quality film and television

Industrial operators

Pre-Production

Visual Concept Design

Composition

Spatial Layering

Human checking

Production

Acting

Pacing

Subject Prominence

Comprehensive Sampling

Test Pair Construction

Cinematography

Lens

Shot-Size Rationality

…

…

Text-to-Video Reference-to-Video

196 Rationales

Aesthetics

Text-to-Multi-Shot-Video Text-to-Video-with-Sound

✓ The shot selection and generation are consistent

Reference

Ground-Truth

Affectivity

with the prompt description.

Prompt

Images and Videos

Videos

✓ The shot size aligns with the intended message and matches the current mood or narrative focus (e.g., using a close-up to convey emotion).

Multi-Shot

Model Test

✓ There are no overly extreme shot sizes that cause the

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

Post-Production

Sound Design

loss of key information. …

Step-III: Expert (Human) Evaluation Step-IV: (Machine) Evaluation Suite

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

Website Professional Operator Development

[Figure 17]

Face Extractor Camera Trajectory Extractor Lighting Extractor

Expression Extractor

Ground

Ranking-Based Scoring

…

Truth Video

(Watch All Videos at Once)

Material Extractor

Aesthetic Evaluator Sentiment Analyzer

Lip-Sync Extractor

&

Chain-of-Thought Evaluation

Reference & Prompt

Thoroughly Considered Scoring

Calibration

Perception Capacity Test

Expert-Guided Multi-Question

[Figure 18]

Reflection

(~2 hours per Pair per Expert)

Task-Specific Reward Model Training

Team Composition

14 Video AIGC Research Scientists and Engineers 20 Professional Artists

###### Data Curation

Preference Data Fine-Tuning

Preference Pair Data Expert Scoring Data

Workflow

Scoring Data Fine-Tuning

Scoring Cross-Checking Validation

Step-V: Application

[Figure 19]

[Figure 20]

Agent System Reward Model for RL Training

Optimized

Planner Tool Call Result Evaluation Original

Reward Singal

RLHF/ GRPO

Benchmarking

Video Model

Video Model

- Figure 1: Overview. EvalVerse systematically digitizes subjective cinematic expertise into a computable, expert-calibrated evaluation framework through five steps. (I) Taxonomy Establishment: Decomposing the professional filmmaking workflow into 3 production stages, encompassing 7 cinematic aspects, 18 main dimensions, 45 sub-dimensions, and 196 granular rationales to structurally define cinematic “goodness.” (II) Dataset Curation: Constructing test pairs across full-modality video generation tasks (e.g., multi-shot, audio-visual) via comprehensive sampling from a millionscale professional database. (III & IV) Expert-Machine Calibration: Bridging the historical divide between human aesthetic perception and algorithmic scoring. By synergizing specialized perception extractors with an expert-guided Chain-of-Thought process, we align Vision-Language Model reasoning with 34 professional experts. (V) Versatile Applications: Beyond static diagnostic benchmarking, EvalVerse serves as a fundamental infrastructure, showing promising potential to provide high-quality reward signals for Reinforcement Learning and act as an expert-level evaluator for autonomous video agent workflows.

### 1 Introduction

The rapid evolution of generative video foundation models OpenAI [2024], Tencent et al. [2024], Google Deepmind [2025b], ByteDance [2026], Kuaishou [2025], Wan et al. [2025] has propelled the field toward a new frontier of cinematic synthesis. Despite achieving remarkable pixel-level visual fidelity through massive Supervised Fine-Tuning (SFT) Jiang et al. [2025], a significant chasm remains between the raw output of these models and the demanding requirements of professional filmmaking. As SFT approaches a scalability bottleneck due to the scarcity of high-quality cinematic data, the field is transitioning toward Reinforcement Learning (RL) paradigms (e.g., RLHF Kaufmann et al. [2023], GRPO Xue et al. [2025]) and agentic workflows Wu et al. [2025] to achieve precise control and complex narratives. In this new era, evaluation is no longer merely a passive leaderboard; it is becoming the critical bottleneck. Professional, reliable, fine-grained evaluation frameworks are therefore the essential prerequisite for providing high-quality reward signals and guiding the next generation of AI-aided cinematic evolution.

However, we observe a critical twofold gap in the current landscape of video generation evaluation. (i) The “Right” vs. “Good” Objective Gap: Existing benchmarks Liu et al. [2024], Huang et al. [2024, 2025], Zheng et al. [2025], Wei et al. [2026] are predominantly stuck in the paradigm of evaluating “whether it is right”—focusing merely on prompt-following capabilities and the basic presence of

Table 1: Comparison of EvalVerse with existing video generation benchmarks. Our framework is the first to achieve full-modality coverage across audio-sync and multi-shot sequencing, while introducing a pipeline-aware paradigm with high interpretability via expert-guided CoT.

Task Modality Coverage Evaluation Paradigm

Benchmark

Text-to-Video Reference-to-Video Video with Sound Multi-Shot Pipeline-Aware Expert-Guided Interpretability

EvalCrafter Liu et al. [2024] ✓ × × × × × Mid VBench Huang et al. [2024] ✓ × × × × × Low VBench 2.0 Zheng et al. [2025] ✓ × × × × × Mid VBench++ Huang et al. [2025] ✓ ✓ × × × × Mid VADB Qiao et al. [2025] × × × × × ✓ High CineTechBench Wang et al. [2025b] × ✓ × × × ✓ Mid Stable Cinemetrics Chatterjee et al. [2025] ✓ × × × Partial ✓ Mid UniVBench Wei et al. [2026] ✓ ✓ × × × × Mid

EvalVerse (Ours) ✓ ✓ ✓ ✓ ✓ ✓ High (CoT)

visual elements. They fundamentally fail to assess “whether it is good,” neglecting the nuanced aesthetic, physical, and cinematic qualities required for professional production. (ii) Methodological and Credibility Gap: The transition from evaluating “rightness” to “goodness” introduces a severe methodological bottleneck. Assessing cinematic quality inherently relies on domain-specific expert knowledge Qiao et al. [2025] and subjective nuances that previous automated metrics fundamentally fail to capture. Consequently, the field is trapped in an evaluation paradox: while professional human assessment is the gold standard, it is prohibitively expensive and unscalable; conversely, generic Vision-Language Models (VLMs)—the default automated alternative Joshi et al. [2026]—lack the professional rigor and domain-specific logic alignment.

To systematically address this twofold gap, we propose EvalVerse (Fig. 1), which takes a pragmatic first step in shifting the evaluation paradigm from generic visual scoring to a structured audit of professional filmmaking. Our framework directly resolves the aforementioned challenges through two corresponding technical contributions:

- (i) Pipeline-Aware Cinematic Taxonomy: To systematically define and measure “goodness,” we propose the first evaluation taxonomy that employs the professional filmmaking workflow as a structured diagnostic lens. Rather than assuming AI generation occurs in discrete steps, we audit the final generated video by mapping its complex multimodal elements back to three traditional production stages: pre-production (assessing foundational visual concept design), production (evaluating dynamic acting, cinematography, aesthetics, & affectivity), and post-production (analyzing multi-shot & sound design). This comprehensive framework captures the nuanced cinematic qualities neglected by previous benchmarks, enabling explainable diagnostic probing of specific model capabilities rather than just outputting a single holistic score.
- (ii) Expert-Calibrated Chain-of-Thought Evaluator: To overcome the evaluation paradox and bridge the credibility gap of automated metrics, we introduce a massive human-in-the-loop calibration process involving professional domain experts (filmmakers and artists), algorithm scientists, and engineers. By repeatedly cross-calibrating human judgments with the actual perceptual and analytical boundaries of current state-of-the-art VLMs Gemini Team, Google [2026], Bai et al. [2025], we develop specialized evaluators that align their internal reasoning logic with professional critics. This pragmatic approach forces the evaluator to generate professional-grade Chain-of-Thought (CoT) rationales before scoring, successfully digitizing subjective, expert-level cinematic knowledge into scalable and interpretable machine metrics. Furthermore, our comprehensive survey (Tab. 1) reveals that existing video benchmarks Liu et al.

- [2024], Huang et al. [2024, 2025], Zheng et al. [2025] significantly lag behind the rapid evolution of foundation models. They Wang et al. [2025b], Wei et al. [2026], Shi et al. [2026], Zhang et al. [2026] predominantly focus on silent, single-shot generation and construct test prompts by artificially permuting isolated cinematic elements Chatterjee et al. [2025], failing to capture authentic cinematic distributions or provide reference videos for evaluation. To address these limitations, EvalVerse incorporates full-modality & multi-shot narrative coverage. Supporting this evaluation is our “Real-to-Gen” data engine for test pair construction, which performs diversified, proportional sampling from real-world professional video datasets. Through hierarchical structural annotation and asset disentanglement, this engine generates high-fidelity test pairs with authentic reference videos, reflecting the true distribution of professional production and eliminating the stochastic bias inherent in existing prompt-based benchmarks.

In summary, EvalVerse treats video evaluation as a core scientific problem—the systematic digitization of subjective cinematic expertise—delivering two key contributions: (i) Methodological Innovation: By organizing domain expertise into a pipeline-aware taxonomy, distilling expert judgments into

a curated dataset, and injecting this knowledge into VLMs via human-machine calibration, we successfully translate abstract professional evaluation into scalable, expert-aligned CoT reasoning. (ii) Comprehensive Coverage & Alignment: EvalVerse retains compatibility with “rightness” and “goodness” while pioneering the evaluation of complex multi-shot sequencing and audio-visual integration, achieving strong human-machine alignment across these advanced dimensions. Looking toward future generative video paradigms, EvalVerse goes beyond a leaderboard by providing trustworthy diagnostic signals, with strong potential to support high-quality reward modeling for Reinforcement Learning and to serve as an expert evaluator for agentic workflows.

### 2 Related Work

#### 2.1 Generative Video Foundation Model

The landscape of generative video foundation models has rapidly advanced from early 3D UNets Blattmann et al. [2023] to scalable DiT Peebles and Xie [2023] and Flow Matching architectures Wan et al. [2025], Tencent et al. [2024]. Beyond architectural scaling, functional capabilities have shifted dramatically. Modern models have evolved from stochastic, silent generation to highly controllable, professional-grade production Yang et al. [2024], Luma AI [2024]. Crucially, recent breakthroughs have successfully introduced end-to-end audio-visual integration OpenAI [2025], HaCohen et al. [2024], Kuaishou [2025], ByteDance [2026] and complex multi-shot narrative sequencing Guo et al. [2025], Meng et al. [2025], Wang et al. [2025a]. This paradigm shift from generating isolated clips to synthesizing cohesive, multimodal cinematic sequences demands entirely new evaluation frameworks.

#### 2.2 Benchmark for Video Generation

Evolution of General Benchmarks: From Consistency to Faithfulness. Early evaluation paradigms primarily relied on holistic metrics such as FVD Unterthiner et al. [2019] and CLIP-Score Radford et al. [2021], which often failed to capture the nuances of temporal dynamics and semantic precision. The landscape shifted with the introduction of VBench Huang et al. [2024], which pioneered the decomposition of video quality into multiple hierarchical dimensions. This was further refined by VBench 2.0 Zheng et al. [2025], which shifted the focus toward intrinsic faithfulness—addressing the misalignment between textual prompts and generated content in complex scenarios. Subsequent iterations Shi et al. [2026], Zhang et al. [2026], Zhou et al. [2026] like VBench++ Huang et al. [2025] expanded the suite’s versatility to cover broader generative capabilities. Simultaneously, UniVBench Wei et al. [2026] attempted to provide a unified evaluation for Video Foundation Models.

Professionalization: Cinematography and Aesthetics. Recognizing that “visual appeal” in professional contexts is governed by cinematographic laws, a new wave of specialized benchmarks has emerged. Stable Cinemetrics Chatterjee et al. [2025] introduced a structured taxonomy for professional video, focusing on the precision of camera control and lighting. CineTechBench Wang et al. [2025b] further narrowed this focus by evaluating a model’s understanding and generation of specific cinematographic techniques. In parallel, the assessment of “beauty” has moved from subjective scoring to multidimensional auditing. VADB Qiao et al. [2025] established a large-scale database with professional-grade annotations for video aesthetics. These works highlight a clear trend: the evaluation for video generation is moving beyond basic prompt-following toward the mastery of the visual language of cinema.

### 3 Taxonomy

The core of EvalVerse is a hierarchical, pipeline-aware taxonomy designed to bridge the gap between AI video synthesis and professional filmmaking standards. Recognizing that modern foundation models typically synthesize videos in an end-to-end manner, we do not assume a multi-step generation process. Instead, we employ the traditional filmmaking workflow as a powerful diagnostic lens. Rather than treating the final generated video as a flat collection of visual attributes, our taxonomy reverse-engineers the assessment by mapping the complex multi-modal elements of the output onto three distinct conceptual stages: Pre-Production, Production, and Post-Production.

###### Visual Concept Design

###### Multi-Shot Rhythm Logic

###### Sound Design Vocal Soundscape

Character

Identification

Acoustic Quality

Scene Consistency Narrative Continuity Spatial Continuity

Shot Duration Rationality Rhythmic Layering

Ambient Sound Fidelity

Costume Rationality

Lip-Sync

Musical Score Alignment Narrative Sound Design

Scene

###### Post-

Environment Plausibility Genre Distinctiveness

Production

###### Acting Consistency Action Face Identity

###### Pre-

Production

###### Expression Accuracy

Action Tension

Cinematography

Facial Tension Expression Diversity

Production

Interaction Plausibility

Attribute

Composition

Action-Emotion Synergy

Continuity

Shot-Size Rationality

Subject Prominence

Aesthetics

Affectivity

Spatial Layering

Grounding

Visual Quality

Materiality

Rendering Quality

###### Pacing

Material Identifiability Stylistic Consistency

Tonal Identifiability Visual-Emotional Synergy

Movement Rationality Lens Depth of Field

Physics

Temporal Consistency

Lighting

Progression

Reference Adherence Lighting Logic Volumetric Sculpting

###### Chromaticity

Transition Continuity Intensity Layering

Focal Length

Harmony Emotive Power

Exposure

Focus

- Figure 2: Pipeline-aware evaluation taxonomy. We propose a comprehensive taxonomy that mirrors the professional cinematic workflow.

#### Table 2: Comparison of evaluation dimensions across the full production pipeline.

Pre-Prod. Prod: Acting Prod: Cinematography Prod: Aesthetics Prod: Affectivity Post-Prod.

Benchmark

Design Acting Composition Lens Pacing Vis. Quality Chromaticity Materiality Lighting Grounding Progression Multi-Shot Sound

EvalCrafter Liu et al. [2024] × × × × × ✓ × × × × × × × VBench Huang et al. [2024] × Partial × × × ✓ × × × × × × × VBench 2.0 Zheng et al. [2025] × Partial × × × ✓ × × × × × × × VBench++ Huang et al. [2025] × Partial × × × ✓ × × × × × × × VADB Qiao et al. [2025] Partial Partial Partial Partial × ✓ ✓ × Partial Partial × × × CineTechBench Wang et al. [2025b] × × ✓ ✓ ✓ × ✓ × ✓ × × × × Stable Cinemetrics Chatterjee et al. [2025] Partial ✓ Partial ✓ Partial × × × Partial Partial × × × UniVBench Wei et al. [2026] × Partial Partial ✓ Partial ✓ Partial × Partial × × × ×

EvalVerse (Ours) ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓

#### 3.1 Pre-Production

This stage evaluates the foundational “Visual Development” and asset design logic before dynamic synthesis occurs. It ensures that the generated assets possess clear identifiability and logical consistency.

#### 3.1.1 Visual Concept Design

As the cornerstone of directing and art design, this dimension audits the conceptual integrity of characters and environments, ensuring they align with the intended worldview and narrative settings.

Character. This dimension audits the foundational asset integrity of the subject. It encompasses Identifiability, which requires clear, recognizable visual anchors (e.g., unique facial structures, body types, and silhouettes) that distinguish the character from others without identity morphing (such as unintended changes in face or clothing). It also includes Costume Rationality, which evaluates whether the character’s attire and styling logically match their intended concept (profession, identity, era), the specific scene context, and the overarching worldview.

Scene. This focuses on the world-building logic of the environment. It includes Environment Plausibility, auditing whether the spatial arrangement of objects follows physical laws (e.g., gravity, collisions, support) and spatial logic (perspective, scale, relations), penalizing AI hallucinations like floating objects or clipping. Furthermore, Genre Distinctiveness measures the purity of the artistic style, ensuring that the visual language (whether realism, animation, or cyberpunk) exhibits clear, characteristic signatures in lighting, materials, and colors, without inappropriate stylistic mixing (e.g., blending 2D and 3D elements illogically).

#### 3.2 Production

This stage evaluates the execution of the “virtual shoot.” It comprehensively assesses how the subject performs, how the camera captures the scene, the overall visual aesthetics, and the emotional atmosphere generated.

#### 3.2.1 Acting

This dimension evaluates the subject’s presentation, focusing on the dynamic consistency, physical kinetic power, and psychological nuance of the character’s performance.

Consistency. This ensures the stability of character assets during movement. It includes Face Identity, requiring facial features to remain consistent across varying angles without morphing or AI-induced structural changes during motion. It also covers Attribute consistency, ensuring that hair length/color, clothing style/material, and accessories remain stable without sudden flickering, disappearing, or unintended transformations.

Action. This evaluates the kinetic power, narrative intent, and physical interactions of movement. It covers Action Tension, ensuring movements follow physical logic (avoiding mechanical or weightless motions) and possess natural kinetic force without biological impossibilities (e.g., bone breaking). It also includes Action-Emotion Synergy, assessing whether the physicality reflects the character’s internal state (e.g., anger driving forceful actions, joy driving lightness) and effectively drives the emotional narrative. Furthermore, it evaluates Interaction Plausibility, ensuring that interactions align with prompt descriptions, demonstrate clear contact and basic force logic, and avoid generation errors like clipping or incorrect positioning, while maintaining logical displacement, movement, and deformation of the interacted objects.

Expression. This assesses the nuance of the character’s facial performance. Metrics include Accuracy (matching the text prompt and contextual logic without contradictory expressions), Facial Tension (natural muscle contractions and micro-expressions, avoiding over-exaggeration or stiffness), Expression Diversity (providing layered, rich emotional changes rather than a monotonous single expression), and Continuity (ensuring smooth, biologically plausible emotional transitions without abrupt jumps).

#### 3.2.2 Cinematography

This dimension evaluates the “virtual camera” language and visual storytelling, auditing how the framing, optical properties, and camera movements serve the narrative.

Composition. This evaluates the framing logic. It includes Shot-Size Rationality (appropriateness of close-ups vs. wide shots for the narrative, avoiding awkward framing like cutting off heads), Subject Prominence (ensuring the main subject is visually salient, not obscured by lighting or messy backgrounds, and effectively guides the viewer’s eye), and Spatial Layering (establishing clear foreground, midground, and background separation, utilizing light and shadow for depth, and maintaining spatial continuity during movement).

Lens. This audits the physical validity of the camera’s optical settings. It encompasses Depth of Field (clear focal planes, natural bokeh gradients, and logical depth changes during movement without edge artifacts), Focal Length (adhering to the perspective logic of wide, standard, or telephoto lenses based on spatial constraints), Focus (clear focus points, logical focus shifts, and tracking, avoiding sudden focus jumps or blurring of key areas), and Exposure (maintaining appropriate dynamic range, matching the scene’s lighting context, and avoiding AI-induced exposure flickering).

Pacing. This evaluates the temporal dynamics of camera movement. It focuses on Movement Rationality, ensuring that camera trajectories (pan, tilt, push, pull) serve a clear narrative purpose, possess appropriate speed and natural kinetic inertia, and are free from unintended AI shaking or aimless drifting.

#### 3.2.3 Aesthetics

This dimension focuses on the technical fidelity and artistic rendering of the video, encompassing visual quality, color grading, physical materiality, and lighting design.

Visual Quality. This focuses on the foundational render fidelity, physical accuracy, and temporal stability of the generated content. Rendering Quality ensures sufficient clarity and high resolving power for distinguishable details. It penalizes visual degradations such as noise, grain, compression artifacts, and edge anomalies (e.g., aliasing or ghosting). Furthermore, it requires rich textural details, avoiding overly smooth or “plastic” appearances, and strictly prohibits generative artifacts like distortions or repetitive textures. Physics evaluates adherence to real-world physical principles, ensuring logical physical morphology and structural details (avoiding shadow, reflection, or structural errors). It ensures objects obey basic physical laws (e.g., gravity, inertia, and material properties), demonstrate plausible force and interaction feedback without weightless or floating effects, and follow rational movement and displacement paths. Finally, Temporal Consistency assesses stability across continuous frames, penalizing fluctuations in clarity, detail flickering or repainting, edge jittering, brightness or color flashes, and sudden local quality degradation (e.g., localized collapse).

Chromaticity. This audits the artistic use of color. It includes Harmony (balanced color grading, unified tones, and absence of abrupt/messy colors) and Emotive Power (how the palette amplifies the intended mood, changes dynamically with the narrative, and utilizes color contrast for visual emphasis).

Materiality. This evaluates surface realism through Material Identifiability (accurate optical properties like reflection, roughness, and transparency to distinguish metal, skin, fabric, or glass, avoiding plastic-looking skin) and Stylistic Consistency (unified shader language across assets that matches the overall lighting and artistic style).

Lighting. This audits the illumination logic. It includes Lighting Logic (matching the prompt’s specified directional/ambient light, time of day, and color temperature, clear light sources, consistent shadow directions and intensities, and absence of unexplained light leaks), and Volumetric Sculpting (how light defines 3D form, spatial depth, and maintains volume dynamically during movement).

#### 3.2.4 Affectivity

This dimension evaluates the emotional resonance and atmospheric setup of the video, ensuring that the visual elements collectively generate a compelling and continuous emotional experience.

Grounding. This assesses the initial atmospheric setup. It includes Tonal Identifiability (establishing a clear emotional baseline, such as neutral, tense, warm, or depressing, that fits the narrative context) and Visual-Emotional Synergy (ensuring color, light, composition, and lens choices collectively express the emotion, avoiding conflicts between visual presentation and the intended mood).

Progression. This evaluates the emotional arc over time. It audits Transition Continuity (smoothness of emotional shifts without causeless, abrupt jumps) and Intensity Layering (the presence of emotional buildup or decrescendo, utilizing visual techniques to enhance emotional peaks, avoiding overly flat or excessively explosive expressions, and adhering to a rhythmic structure of setup, development, and climax).

#### 3.3 Post-Production

The final stage evaluates the assembly of shots and the multimodal integration, focusing on multi-shot logic and audio-visual synchronization. While these evaluation dimensions could theoretically extend to traditional editing and dubbing, assessing complex post-processing interventions remains highly challenging. Therefore, our current scope strictly focuses on natively generated multi-shot sequences and synthesized audio.

#### 3.3.1 Multi-Shot

This dimension evaluates the sequential logic and temporal rhythm between multiple shots, ensuring narrative flow and spatial coherence.

Logic. This evaluates sequential continuity. Metrics include Scene Consistency (stable environments, props, lighting, weather, and character makeup across cuts without AI generation errors), Narrative Continuity (logical cause-and-effect action sequencing and subject state maintenance), and Spatial Continuity (adherence to character positioning, the 180-degree rule, and spatial orientation).

Sampling

Industrial

Human

Database

Operators

Checking

Hyper-real 15%

Romantic

Fine-motor 15%

Positive 10%

10%

Old 5%

Warm 10%

Epic 15%

Annotation Metadata

TV 20%

Costume

Negative 10%

Bright

Fighting

10%

Child 5%

{ ”global_context": { "type": { "main_category": "Anime",

15%

25%

Age

3D 15%

Emotion

Nature 15%

Action 40%

Lighting/ Mood 10%

Anime

10%

Type

Scene 10%

30%

Joyful 10%

Mood 50%

"sub_category": "3D" },

45%

5%

Actor

Male 10%

Neutral 10%

"era": "Ancient", "fantasy_genre": "Chinese Magic" },

15%

Film 35%

Gender

Dark 25%

Drama 40%

20%

Full-

Horror

2D 15%

Body

"cinematography": {

10%

15%

Action

Female 10%

"shot_details": {

Somber 10%

30%

"is_multi_shot": true, "dominant_shot_size": "Medium", "shot_sequence": [

complete Case(Preview)In

Medium 45%

Wide 20%

Sci-fi

{ "shot_size": "Wide", "duration_sec": 2.5 },

30%

Nature

30%

{ "shot_size": "Medium",

Fantasy Genres

Chinese

Vocal 30%

Era

Audio 15%

Shot Size 10%

Multi-shot

Magic

"duration_sec": 4.0 } ] }, "lighting_mood": {

Magic

50%

10%

15%

Ancient

50%

10%

30%

Close-up 35%

Modern 55%

"primary_attribute": "Mood",

Non-vocal SFX 30%

"specific_mood": "Somber" } },

"content_semantics": { "scene": {

###### Test Pair Construction

ControlNet

2. Subject

3. Object Task

"primary_category": "Action", "sub_category": "Fighting" }, "actor_metadata": {

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

✓ Text-to-Video: Prompt

Ground Truth Video Metadata

✓ Text-to-Multi-Shot-Video: Prompt

[Figure 26]

"has_human_actor": true, "actors": [

4. Scene

✓ Text-to-Video-with-Sound: Prompt

{ "actor_id": "char_01",

[Figure 27]

"is_main_character": true,

1. Prompt

"gender": "Male", "age_group": "Child", "emotion": "Negative", “costume_tags”: [“traditional robe”], “action_tags": ["sword fighting", ] } ] } },

✓ Subject-to-Video: Prompt, Subject

This is a realistic film clip depicting an indoor drama scene. The

sequence features a multi-shot edit, alternating between medium and close-up shots, while the dim, warm Lighting establishes a distinctly Somber Mood. The scene focuses on a female adult actor exhibiting a neutral emotion. Dressed in an elegant floral

✓ Object-to-Video: Prompt, Object

5. Depth Sequence

[Figure 28]

✓ Scene-to-Video: Prompt, Scene

[Figure 29]

Qipao, her primary action involves drinking from a glass and gazing thoughtfully out the window. The expected audio consists of subtle ambient noise.

"audio_track": { "has_audio": true, "audio_elements": ["Vocal"] } }

✓ Camera-to-Video: Prompt, Depth

- Figure 3: Comprehensive pipeline for dataset annotation, sampling, and test pair construction. (Left) The annotation pipeline, yielding structured JSON metadata via industrial operators and human verification. (Top Right) Proportional distributions ensuring balanced and comprehensive data sampling. (Bottom) Test pair construction generating multi-modal inputs for diverse downstream generation tasks.

Rhythm. This audits the temporal heartbeat of the edit. It includes Shot Duration Rationality (providing sufficient time for information consumption, matching the emotional tension and audio rhythm) and Rhythmic Layering (varying cutting tempos, combining dynamic and static shots, and aligning the rhythm with the narrative arc from setup to climax).

3.3.2 Sound Design

This dimension evaluates the relationship between sound and image, auditing the integration of human voices, ambient sounds, and musical scores.

Vocal. This evaluates human voice integration. It includes Acoustic Quality (matching the character’s age, gender, and personality, ensuring technical purity without mechanical noise, and maintaining spatial reverb consistency) and Lip-Sync (temporal alignment between phonemes and mouth shapes, matching mouth opening with volume, and aligning vocal emotion with facial expression), and Narrative Sound Design (using off-screen audio cues to convey emotion, guide audience attention, and expand the narrative space).

Soundscape. This focuses on the immersive sonic environment. It audits Ambient Sound Fidelity (realism of Foley, spatial depth, and matching the on-screen environment/weather) and Musical Score Alignment (synchronizing musical tone and rhythm with visual cuts and emotional beats without overpowering dialogue).

- 4 Dataset Curation: Test Pair Construction

To capture professional filmmaking complexities in the EvalVerse benchmark, our data engine (Fig. 3) transforms raw cinematic videos into “Real-to-Gen” test pairs via structured annotation, strategic sampling, and test pair construction.

Annotation. Starting with a diverse database of professional films and animations, we employ a multimodal perception suite to extract structured metadata covering our entire evaluation taxonomy (e.g., camera parameters, character attributes, and environments). Following industrial-grade processing

|[Figure 30]<br><br>[Figure 31]<br><br>[Figure 32]<br><br>[Figure 33]<br><br>[Figure 34]<br><br>[Figure 35]<br><br>[Figure 36]<br><br>[Figure 37]<br><br>[Figure 38]<br><br>[Figure 39]<br><br>[Figure 40]<br><br>[Figure 41]<br><br>[Figure 42]<br><br>[Figure 43]<br><br>[Figure 44]<br><br>[Figure 45]<br><br>[Figure 46]<br><br>[Figure 47]<br><br>[Figure 48]<br><br>[Figure 49]<br><br>[Figure 50]<br><br>[Figure 51]<br><br>[Figure 52]<br><br>[Figure 53]<br><br>[Figure 54]<br><br>[Figure 55]<br><br>[Figure 56]<br><br>[Figure 57]<br><br>[Figure 58]<br><br>[Figure 59]<br><br>[Figure 60]<br><br>[Figure 61]<br><br>[Figure 62]<br><br>[Figure 63]<br><br>[Figure 64]<br><br>[Figure 65]<br><br>[Figure 66]<br><br>[Figure 67]<br><br>[Figure 68]<br><br>[Figure 69]<br><br>[Figure 70]<br><br>[Figure 71]<br><br>[Figure 72]<br><br>[Figure 73]<br><br>[Figure 74]<br><br>[Figure 75]<br><br>[Figure 76]<br><br>[Figure 77]<br><br>[Figure 78]<br><br>[Figure 79]<br><br>[Figure 80]<br><br>[Figure 81]<br><br>[Figure 82]<br><br>[Figure 83]<br><br>[Figure 84]<br><br>[Figure 85]<br><br>[Figure 86]<br><br>[Figure 87]<br><br>[Figure 88]<br><br>[Figure 89]<br><br>[Figure 90]<br><br>[Figure 91]<br><br>[Figure 92]<br><br>[Figure 93]<br><br>0.000000<br><br>0.100000<br><br>0.200000<br><br>0.300000<br><br>0.400000<br><br>0.500000<br><br>0.600000<br><br>0.700000<br><br>0.800000<br><br>0.900000<br><br>Visual concept design Acting Cinematography Aesthetics Affectivity Multi-Shot cutting Sound design<br><br>Seedance 2.0 Kling-v3-Omni Happy Horse 1.0 Hailuo 2.3 Vidu-Q2-Pro Wan 2.2 Hunyuan 1.5 LTX2 HoloCine UniVideo MultiShotMaster<br><br>[Figure 94]<br><br>[Figure 95]<br><br>[Figure 96]<br><br>[Figure 97]<br><br>[Figure 98]<br><br>[Figure 99]<br><br>[Figure 100]<br><br>[Figure 101]<br><br>[Figure 102]<br><br>[Figure 103]<br><br>[Figure 104]<br><br>[Figure 105]<br><br>[Figure 106]<br><br>[Figure 107]<br><br>[Figure 108]<br><br>[Figure 109]<br><br>[Figure 110]<br><br>[Figure 111]<br><br>[Figure 112]<br><br>[Figure 113]<br><br>[Figure 114]<br><br>[Figure 115]<br><br>[Figure 116]<br><br>[Figure 117]<br><br>[Figure 118]<br><br>[Figure 119]<br><br>[Figure 120]<br><br>[Figure 121]<br><br>[Figure 122]<br><br>[Figure 123]<br><br>[Figure 124]<br><br>[Figure 125]<br><br>[Figure 126]<br><br>[Figure 127]<br><br>[Figure 128]<br><br>[Figure 129]<br><br>[Figure 130]<br><br>[Figure 131]<br><br>[Figure 132]<br><br>[Figure 133]<br><br>[Figure 134]<br><br>[Figure 135]<br><br>[Figure 136]<br><br>[Figure 137]<br><br>[Figure 138]<br><br>[Figure 139]<br><br>[Figure 140]<br><br>[Figure 141]<br><br>[Figure 142]<br><br>[Figure 143]<br><br>[Figure 144]<br><br>[Figure 145]<br><br>[Figure 146]<br><br>[Figure 147]<br><br>[Figure 148]<br><br>[Figure 149]<br><br>[Figure 150]<br><br>[Figure 151]<br><br>[Figure 152]<br><br>[Figure 153]<br><br>[Figure 154]<br><br>[Figure 155]<br><br>[Figure 156]<br><br>[Figure 157]|
|---|

#### Figure 4: Overall performance comparison of evaluated video generation models.

and rigorous manual verification, these highly accurate labels serve as robust ground-truth metadata for downstream sampling and prompt generation.

Sampling. To ensure the benchmark is both comprehensive and industry-representative, we perform diversified sampling from the annotated database. Rather than a stochastic selection, we adopt a proportional sampling strategy across nine core cinematic dimensions to maintain a balanced distribution.

Construction. This stage involves the construction of multi-modal test pairs tailored for different generation tasks. We utilize a Gemini 3.1 Pro Gemini Team, Google [2026] to ingest the structured metadata and raw video captions, synthesizing professional-grade test prompts that reflect cinematic terminology. For reference-based tasks (e.g., subject-driven generation), we extract keyframes from the source videos and employ Nano Banana Pro Google Deepmind [2025a] to generate high-fidelity reference images. For depth reference, we adopt ControlNet-tuned Zhang et al. [2023] model to generate depth sequences.

### 5 Benchmark: Expert Evaluation Results

- 5.1 Benchmarking Settings

Human Evaluation Protocol. To guarantee both cinematic aesthetics and algorithmic fidelity, our evaluation is conducted by a multi-disciplinary team (filmmakers, AIGC scientists, and engineers) through a strict three-stage pipeline: (i) Discriminative Annotation: Annotators perform side-by-side comparisons of the prompt, Ground Truth video, and model outputs. Crucially, to yield strong preference signals, they must assign strict discriminative rankings across all predefined dimensions. (ii) Quality Assurance: Senior film industry professionals conduct item-by-item reviews of the initial annotations, assigning Pass/Fail labels based on cinematic validity and consistency. (iii) Final Audit: Experts oversee the ultimate verification, resolving anomalies and eliminating systemic bias to finalize the ground-truth human preference dataset.

Video Generation Model Selection. (i) Closed-Source Models: We include Kling-v3Omni Kuaishou [2025], Seedance 2.0 ByteDance [2026], Happy Horse 1.0 Team [2026], Vidu-Q2Pro Vidu [2024], and Hailuo 2.3 MiniMax [2026]. (ii) Open-Source Models: We evaluate Hunyuan

- 1.5 (8.3B) Tencent et al. [2024], LTX2 (19B) HaCohen et al. [2024], and Wan2.2 (14B) Wan et al. [2025]. (iii) Multi-Shot and Audio-Visual Models: We specifically select models that push the boundaries of multimodal cinematic synthesis. On one hand, models like HoloCine (14B) Meng et al. [2025] and MultiShotMaster (14B) Wang et al. [2025a] are at the frontier of multi-shot narrative sequencing. On the other hand, models such as Kling-v3-Omni, Seedance 2.0, Happy Horse 1.0, and LTX2 feature native sound design capabilities.

5.2 Benchmarking Analysis Overall. As shown in Fig. 4, the evaluated models exhibit a clear hierarchical distribution. Seedance

- 2.0 achieves the best comprehensive performance, demonstrating consistently strong results in every way. Kling-v3-Omni and Happy Horse 1.0 constitute the next leading group: the former shows strong and stable performance in aesthetics, cinematography, visual concept preservation, and sound-related dimensions, while the latter performs particularly well in cinematography, visual concept design, acting, and multi-shot organization. In comparison, Hailuo 2.3 and Vidu-Q2-Pro form a competitive middle tier, with strengths mainly concentrated in cinematography, aesthetics, and visual concept

###### AESTHETICS

###### ACTING

###### AFFECTIVITY

Action: Action-

Chromaticity:

Grounding: Tonal

Emotion Synergy

Emotive Power

Identifiability

Visual Quality:

Expression: Facial Tension

Action: Action Tension

Chromaticity: Harmony

Temporal Consistency

###### VISUAL CONCEPT DESIGN

Action:

Expression: Diversity

Visual Quality: Rendering Quality

Lighting: Lighting Logic

Interaction

Progression:

Grounding:

Character:

Rationality

Transition Continuity

Visual-Emotional Synergy

Costume

Rationality

Lighting:

Expression:

Consistey:

Visual Quality:

Volumetric Sculpting Materiality: Material Identifiability

Continuity

Attribute

Physics

Materiality: Stylistic Consistency

Expression:

Consistency:

Accuracy

Face Identity

Progression:

Scene: Genre

Character:

Intensity Layering

Distinctiveness

Identifiability

###### MULTI-SHOT

###### CINEMATOGRAPHY

SOUND DESIGN

Composition: Shot-Size Rationality

Soundscape: Ambient Sound Fidelity

Logic: Narrative Continuity

Scene:

Environment

Plausibility

Pacing: Movement Rationality

Composition: Spatial Layering

Hailuo 2.3 HoloCine

Rhythm: Shot

Soundscape:

Logic: Scene

Duration

Vocal: Lip-Sync

Musical Score

Wan 2.2 Kling-v3-Omni

Consistency

Rationality

Alignment

Composition:

LTX2 MultiShotMaster

Lens: Focus

Subject

Seedance 2.0 Hunyuan 1.5

Prominence

Happy Horse 1.0

Lens: Focal Length

Lens: Depth of Field Lens: Exposure

Rhythm: Rhythmic

Logic: Spatial

Vocal: Acoustic

Vocal: Narrative

Layering

Continuity

Quality

Sound Design

#### Figure 5: Fine-grained performance comparison of evaluated models in the Text-to-Video (T2V) setting.

AESTHETICS

ACTING

AFFECTIVITY

Action: Action-

Chromaticity:

Grounding: Tonal

Emotion Synergy

Emotive Power

Identifiability

Visual Quality:

Expression: Facial Tension

Action: Action Tension

Chromaticity: Harmony

Temporal Consistency

VISUAL CONCEPT DESIGN

Action:

Expression: Diversity

Visual Quality: Rendering Quality

Lighting: Lighting Logic

Interaction

Character: Costume

Progression:

Grounding:

Rationality

Transition Continuity

Visual-Emotional Synergy

Rationality

Lighting:

Expression:

Consistey:

Visual Quality:

Volumetric Sculpting Materiality: Material Identifiability

Continuity

Attribute

Physics

Materiality: Stylistic Consistency

Expression:

Consistency:

Accuracy

Face Identity

Progression:

Scene: Genre

Character:

Intensity Layering

Distinctiveness

Identifiability

MULTI-SHOT

CINEMATOGRAPHY

SOUND DESIGN

Composition: Shot-Size Rationality

Soundscape: Ambient Sound Fidelity

Logic: Narrative Continuity

Scene:

Environment

Plausibility

Pacing: Movement Rationality

Composition: Spatial Layering

Kling-v3-Omni

Seedance 2.0

Kling3-Omni Seedance2.0

Rhythm: Shot

Soundscape:

UniVideo Vidu-Q2-Pro

Logic: Scene

Duration

Vocal: Lip-Sync

Musical Score

Consistency

Rationality

Alignment

Composition:

Happy Horse

Happy Horse 1.0

Lens: Focus

Subject

Prominence

Lens: Focal Length

Lens: Depth of Field Lens: Exposure

Rhythm: Rhythmic

Logic: Spatial

Vocal: Acoustic

Vocal: Narrative

Layering

Continuity

Quality

Sound Design

#### Figure 6: Fine-grained performance comparison of evaluated models in the Reference-to-Video (R2V) setting.

design, but relatively weaker performance in affectivity and sound-related dimensions. Wan2.2, Hunyuan 1.5, and LTX2 show moderate overall capability, with advantages mainly in visual and camera-related criteria, whereas HoloCine, UniVideo, and MultiShotMaster present more uneven or specialized performance profiles.

Text-to-Video. As shown in Fig. 5, Seedance 2.0 remains the strongest overall model, achieving the highest average score and consistently ranking near the top across most fine-grained criteria. Its performance is particularly strong in soundscape fidelity, identity preservation, attribute consistency, visual quality, and camera control, indicating robust semantic preservation and balanced perceptual quality. Kling-v3-Omni also maintains a leading position, with strong results in visual concept design, aesthetics, cinematography, and attribute consistency, though its sound-design-related scores are relatively less dominant. Happy Horse 1.0 emerges as a highly competitive newly evaluated model, showing strong performance in chromatic harmony, narrative continuity, subject prominence, visual concept design, and cinematography, while remaining weaker in tonal grounding and narrative sound design. The remaining models show more specialized or uneven profiles. Hailuo 2.3 and Wan2.2 form a competitive middle tier: Hailuo 2.3 performs well in lighting logic, temporal consistency, and camera-related criteria, while Wan2.2 shows strengths in visual consistency and concept preservation but is weaker in action and expression-related dimensions. Hunyuan 1.5 demonstrates moderate visual generation quality, especially in attribute consistency and lighting-related criteria, but its action interaction and expression dimensions are less competitive. LTX2 performs reasonably in lighting and lens-related criteria, yet shows limitations in expression, rhythm, and audio-visual synchronization.

HoloCine presents localized strengths in scene plausibility and multi-shot consistency, but lower acting and rendering-related scores limit its overall quality. MultiShotMaster performs relatively well in shot-duration rationality while lagging behind in broader visual concept design, acting, aesthetics, and sound dimensions.

Reference-to-Video. As shown in Fig. 6, Seedance 2.0, Kling-v3-Omni, and Happy Horse 1.0 form the leading group. Seedance 2.0 achieves the highest overall score and maintains strong performance across most evaluated tertiary dimensions, with particularly strong results in vocal acoustic quality, chromatic harmony, attribute consistency, visual concept preservation, and camerarelated criteria. Kling-v3-Omni ranks second overall and remains highly competitive, performing strongly in vocal acoustic quality, chromatic harmony, ambient sound fidelity, lens focus, and multishot scene consistency, although its action tension and expression-related scores are relatively less dominant. Happy Horse 1.0 also shows a stable and competitive profile, with strengths in subject prominence, chromatic harmony, lighting logic, visual concept design, and cinematography, but weaker results in multi-shot rhythm, lip-sync, and narrative sound design. Vidu-Q2-Pro forms a moderate performance tier. It performs well in lighting logic, chromatic harmony, focal-length control, costume rationality, and subject prominence, but shows weaker results in action interaction, physical realism, movement pacing, and narrative sound design. UniVideo exhibits the most uneven R2V performance: it retains reasonable scores in static visual attributes such as lighting logic, focal length, genre distinctiveness, spatial continuity, and shot-duration rationality, but falls behind on acting, expression dynamics, action-emotion synergy, affective progression, and sound-related dimensions.

### 6 Machine Evaluation Suite

The ultimate objective of our framework is to mathematically model human expert annotations, denoted as H. Formally, given a generated video V ∈ RT×H×W×C, audio A, text prompt p, and reference r, our framework computes a multi-dimensional cinematic score vector S ∈ RD to approximate expert judgment: S ≈ H(V,A,p,r). To achieve this, we design a systematic evaluation pipeline (Sec. 6.1) powered by a fine-tuned VLM (Sec. 6.2).

#### 6.1 Expert-Calibrated Evaluation Pipeline

During the inference phase, our evaluation pipeline operates through a two-step mechanism: extracting deterministic evidence via specialized operators, followed by step-by-step reasoning using the finetuned VLM guided by expert multi-questioning.

#### 6.1.1 Professional Operator Extraction (Perception Prior)

VLMs inherently struggle with fine-grained temporal tracking and low-level perception. To mitigate hallucinations and provide reliable contextual priors, we first deploy a suite of specialized operators Φ = {ϕ1,...,ϕK} to extract deterministic, objective evidence Eprof:

K

ϕk(V,A,p,r), (1)

Eprof =

k=1

where Φ includes operators such as DINO Oquab et al. [2023] and InsightFace Deng et al. [2019] for cross-frame identity tracking, YOLO Khanam and Hussain [2024] for semantic anchoring, SyncNet Chung and Zisserman [2016] for audio-visual synchronization, and Whisper Radford et al. [2023] for speech emotion recognition.

#### 6.1.2 Expert-Guided CoT Reasoning & Scoring

Equipped with the perception prior Eprof, the core evaluation is performed by our fine-tuned VLM, denoted as Mθ∗ (training details in Sec. 6.2). Let X = (A,p,r,Eprof,Q) denote the comprehensive multi-modal context, where Q represents expert-designed multi-questions for a specific cinematic dimension. Rather than outputting a direct score blindly, Mθ∗ performs step-by-step reasoning, generating a detailed CoT.

Crucially, this reasoning phase incorporates a Self-Reflection mechanism within the CoT, forcing the VLM to take a step back and re-examine whether its judgments and reasoning have been subject to

any hallucinations. Furthermore, we introduce a Context-Aware Gating mechanism, represented by an indicator function Igate(p,C) ∈ {0,1}, which dynamically bypasses specific metrics (e.g., strong expressions) if the narrative context C does not warrant them. The final score Sd for dimension d is computed as:

Sd = Mθ∗(V,X) · Igate(p,C). (2)

#### 6.2 Two-Stage VLM Fine-Tuning for Human Alignment

To equip the foundational VLM Mθ with professional cinematic judgment and the aforementioned reasoning capabilities, we fine-tune it through a two-stage paradigm using our curated expert dataset.

#### 6.2.1 Preference Alignment

We first train the model on a large-scale dataset of pairwise comparisons, Dpref = {(Vw,Vl,X)}, where Vw and Vl are the preferred (win) and rejected (lose) videos, respectively. In this stage, the model learns relative cinematic aesthetics and human preferences by minimizing a Bradley-Terry ranking loss:

[log σ (Mθ(Vw,X) − Mθ(Vl,X))], (3) where σ is the sigmoid function.

Lpref(θ) = −ED

pref

#### 6.2.2 Score Calibration

To map these relative preferences into absolute, interpretable metrics and instill CoT reasoning capabilities, we subsequently fine-tune the model on a pointwise dataset Dscore = {(Vi,Xi,Zi,yd,i)}, where Zi is the ground-truth expert CoT and yd,i is the absolute expert score. The model is trained to autoregressively generate the rationale Zi followed by the final score yd,i. The optimal parameters θ∗ are obtained by minimizing the Cross-Entropy loss LCE:

θ∗ = arg min

[LCE (Mθ(V,X),(Z,yd))]. (4)

##### ED

score

θ

### 7 Human-Machine Calibration

#### 7.1 Progressive Calibration Mechanism

To bridge the gap between demanding cinematic expert criteria and the perceptual limits of current VLMs, we propose a progressive, three-tiered calibration mechanism: (i) Prompt-Level (Rationale Replacement): Through iterative calibration, we explicitly replace evaluation dimensions and multiquestions that are overly abstract or beyond the model’s perception and reasoning capabilities. (ii) Fusion-Level (Weight Optimization): To determine the exact score proportions of individual multiquestions, operator evidence (Eprof), and VLM perceptual results within the CoT, we employ a data-driven weight optimization trick. A lightweight MLP trained on human annotations to learn different weights, mitigating operator Out-of-Domain failures and VLM reasoning errors. (iii) Parameter-Level (Knowledge Injection): Fine-tuning the VLM on our expert dataset explicitly injects cinematic domain knowledge, transforming a general VLM into a specialized, expert-aligned reward model.

#### 7.2 Alignment Analysis

To rigorously validate the efficacy of our three-tiered calibration mechanism, we systematically assess the human-machine alignment between EvalVerse and our expert panel from three complementary perspectives: (i) granular win-ratio comparison (Tab. 3), (ii) statistical correlation analysis (Tab. 4), and (iii) trend consistency visualization (Fig. 7). Across all evaluations, we adopt the pairwise win-ratio Huang et al. [2024, 2025] as the unified comparison signal. Specifically, for every candidate model and sub-dimension, we compute its win-ratio against all competitors and measure the correlation between human-derived and EvalVerse-predicted scores. A higher correlation (ρ) indicates that our evaluator faithfully reproduces the relative ordering of professional human preferences.

- Table 3: Human-machine alignment: pairwise win ratios. For each video generation model and evaluation dimension, we report the pairwise win ratio against all other competitors, formatted as “Machine Win Ratio (left) / Human Win Ratio (right)”. The consistent correspondence between our automated predictions and expert annotations validates the efficacy of our expert-calibrated evaluation pipeline.

Evaluation Dimensions Seedance 2.0 Kling-v3-Omni Happy Horse 1.0 HoloCine

Master

MultiShot

LTX2 Hailuo 2.3 Hunyuan 1.5 Wan2.2 UniVideo Vidu-Q2-Pro Visual Concept Design

Character 0.61/0.63 0.47/0.68 0.74/0.82 0.25/0.28 0.25/0.05 0.38/0.05 0.48/0.89 0.56/0.61 0.42/0.39 0.37/0.12 0.61/0.56 Scene 0.53/0.78 0.53/0.61 0.82/0.73 0.62/0.53 0.22/0.05 0.47/0.20 0.65/0.64 0.37/0.44 0.52/0.34 0.19/0.07 0.65/0.55

Consistency 0.79/0.81 0.53/0.74 0.58/0.66 0.48/0.48 0.22/0.20 0.05/0.05 0.90/0.70 0.50/0.20 0.47/0.40 0.52/0.12 0.39/0.29 Acting Action 0.65/0.75 0.48/0.65 0.64/0.72 0.32/0.35 0.13/0.18 0.33/0.05 0.63/0.33 0.16/0.27 0.83/0.67 0.12/0.10 0.68/0.53

Expression 0.58/0.70 0.44/0.62 0.62/0.68 0.24/0.32 0.16/0.16 0.42/0.32 0.42/0.44 0.66/0.58 0.43/0.47 0.42/0.30 0.57/0.40 Cinematography

Composition 0.61/0.72 0.54/0.69 0.72/0.80 0.81/0.56 0.24/0.12 0.36/0.05 0.74/0.89 0.31/0.18 0.76/0.47 0.37/0.26 0.46/0.36 Pacing 0.81/0.75 0.33/0.58 0.78/0.68 0.75/0.50 0.05/0.05 0.60/0.20 0.90/0.90 0.80/0.40 0.20/0.05 0.63/0.27 0.45/0.41 Lens 0.74/0.76 0.48/0.67 0.68/0.59 0.58/0.44 0.05/0.05 0.46/0.16 0.45/0.29 0.44/0.39 0.57/0.56 0.40/0.12 0.44/0.46 Visual Quality 0.71/0.66 0.66/0.84 0.68/0.78 0.44/0.33 0.05/0.05 0.67/0.33 0.53/0.77 0.53/0.27 0.33/0.20 0.28/0.14 0.31/0.28 Chromaticity 0.73/0.76 0.64/0.76 0.72/0.60 0.39/0.38 0.13/0.05 0.77/0.80 0.23/0.05 0.63/0.40 0.37/0.80 0.22/0.13 0.38/0.35 Lighting 0.70/0.75 0.50/0.74 0.68/0.76 0.67/0.50 0.05/0.05 0.40/0.05 0.75/0.60 0.35/0.25 0.45/0.60 0.34/0.06 0.46/0.38

Aesthetics

Materiality 0.81/0.74 0.65/0.77 0.74/0.64 0.51/0.33 0.05/0.05 0.55/0.10 0.58/0.68 0.47/0.43 0.35/0.23 0.23/0.11 0.53/0.47 Affectivity

Grounding 0.57/0.68 0.54/0.62 0.82/0.78 0.54/0.58 0.06/0.05 0.54/0.42 0.56/0.86 0.54/0.48 0.54/0.38 0.54/0.40 0.54/0.50 Progression 0.55/0.55 0.53/0.70 0.72/0.62 0.53/0.65 0.40/0.15 0.48/0.28 0.60/0.85 0.42/0.22 0.53/0.60 0.48/0.32 0.46/0.28

Logic 0.40/0.36 0.75/0.69 0.80/0.88 0.45/0.75 0.25/0.20 -/- -/- -/- -/- -/- -/Multi-Shot

Rhythm 0.50/0.68 0.75/0.65 0.85/0.82 0.40/0.62 0.20/0.05 -/- -/- -/- -/- -/- -/Sound Design

Vocal 0.45/0.58 0.60/0.72 0.85/0.72 -/- -/- 0.35/0.40 -/- -/- -/- -/- -/Soundscape 0.55/0.58 0.35/0.55 0.72/0.78 -/- -/- 0.35/0.30 -/- -/- -/- -/- -/-

- Table 4: Human-machine alignment: correlation coefficients. We report the Spearman Rank Correlation Coefficient (SRCC) and Pearson Linear Correlation Coefficient (PLCC), along with their respective p-values, between EvalVerse and human expert evaluations across all fine-grained dimensions. The consistently high correlation scores demonstrate that our automated metrics robustly align with professional human perception.

Evaluation Dimensions Model Number SRCC psrcc PLCC pplcc Visual Concept Design

Character 11 +0.7529 0.0075 +0.7664 0.0059

Scene 11 +0.8082 0.0026 +0.8224 0.0019 Consistency 11 +0.7472 0.0082 +0.7736 0.0052

Acting Action 11 +0.7636 0.0062 +0.7949 0.0035

Expression 11 +0.8276 0.0017 +0.7872 0.0040 Cinematography

Composition 11 +0.7545 0.0073 +0.8119 0.0024 Pacing 11 +0.7517 0.0076 +0.7406 0.0091 Lens 11 +0.8018 0.0030 +0.7899 0.0038 Visual Quality 11 +0.7991 0.0032 +0.7875 0.0040 Chromaticity 11 +0.7460 0.0084 +0.8067 0.0027 Lighting 11 +0.8174 0.0021 +0.7840 0.0043

Aesthetics

Materiality 11 +0.8091 0.0026 +0.8246 0.0018 Affectivity

Grounding 11 +0.8318 0.0015 +0.7996 0.0031 Progression 11 +0.8457 0.0010 +0.7634 0.0063

Logic 5 +0.9000 0.0374 +0.8430 0.0729 Multi-Shot

Rhythm 5 +0.9000 0.0374 +0.8300 0.0820 Sound Design

Vocal 4 +0.9487 0.0513 +0.8460 0.1540 Soundscape 4 +0.9487 0.0513 +0.8502 0.1498

#### 7.2.1 Alignment Results

We present a comprehensive analysis of our alignment results across the three aforementioned perspectives. First, Tab. 3 details the granular pairwise win-ratios, revealing a striking absolute proximity between EvalVerse predictions and expert annotations across all candidate models. Building upon this raw data, Tab. 4 reports the per-dimension Spearman Rank Correlation Coefficient (SRCC) and Pearson Linear Correlation Coefficient (PLCC). Across all sub-dimensions, ρ stays within a tight band. Finally, Fig. 7 visualizes these relationships through scatter plots and linear regressions, where the tight linear fits further corroborate the robustness of our automated metrics.

More importantly, the statistical results in Tab. 4 trace out a clear pattern that is highly consistent with our design intuition: (i) Pixel-grounded dimensions (Visual Concept Design, Cinematography, Acting, Aesthetics, Affectivity), which are covered primarily by prompt-level CoT, attain strong alignment. This demonstrates that CoT-based digitization already provides a reliable backbone for the majority of cinematic criteria. (ii) Abstract and temporally-entangled dimensions, which are additionally calibrated by task-specific SFT, deliver the highest agreement with human experts. This directly verifies that parameter-level knowledge injection is the decisive step for the hardest dimensions, where natural-language rationales alone cannot close the human-machine gap.

#### 7.2.2 Discussion: The Complementary Synergy of CoT and SFT

While prompt-level CoT on frozen VLMs effectively digitizes perceptually grounded dimensions (e.g., lighting, chromaticity) to provide broad, interpretable alignment, it fundamentally hits a perceptual ceiling for subjective, temporally-entangled, or cross-modal aspects (e.g., multi-shot rhythm). Abstract concepts like “rhythmic layering” cannot be robustly decomposed into zero-shot observable tokens, regardless of prompt elaboration. To overcome this limitation of purely verbalized

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

- Figure 7: Human-machine alignment: visualizing consistency. Each plot correlates expert (x-axis) and machine (y-axis) win ratios per model. Linear fits and Pearson’s ρ confirm that EvalVerse strongly aligns with human judgment across all dimensions.

prompts, we introduce task-specific SFT as a complementary calibration tier. By explicitly injecting the human scoring distribution directly into the VLM’s parameters, SFT supplies the critical lastmile alignment exactly where abstract cinematic expertise lives. Rather than competing, these two paradigms synergize: CoT ensures transparent reasoning across the pipeline, while SFT bridges the perception-reasoning gap for complex dimensions.

- 8 Conclusion

In this work, we introduced EvalVerse, fundamentally redefining video generation assessment from basic prompt-following (“whether it is right”) to a rigorous audit of professional filmmaking (“whether it is good”). By structurally mirroring the real-world pipeline and proposing a systematic human-machine calibration mechanism, we provide a principled framework for characterizing and injecting nuanced human preferences into algorithmic scoring. This successfully digitizes subjective expertise into computable metrics, bridging the long-standing credibility gap between human aesthetic perception and machine evaluation. Extending far beyond a static leaderboard, EvalVerse establishes a foundational infrastructure for the post-SFT era by supplying dense, expert-aligned reward vectors for Reinforcement Learning and explainable diagnostic feedback for autonomous agentic workflows. By providing this critical “missing link,” it catalyzes the transformation of generative models from passive clip generators into professional-grade virtual directors, ushering in a new era of computable cinematography for computer graphics community.

Limitations and Future Work. Future work will address several key challenges: (i) VLM Bottlenecks: Current VLMs process discrete keyframes rather than continuous streams, limiting its temporal perception; (ii) Long-Form Narratives: Scaling evaluation to macro-narratives (e.g., 10+ minutes) requires advanced long-context reasoning; and (iii) Artistic Diversity: Assessing boundless avant-garde styles remains difficult. Ultimately, natively integrating “evaluation” as a fundamental “understanding” task into unified multi-modal models represents a highly promising frontier.

### References

Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, Wenbin Ge, Zhifang Guo, Qidong Huang, Jie Huang, Fei Huang, Binyuan Hui, Shutong Jiang, Zhaohai Li, Mingsheng Li, Mei Li, Kaixin Li, Zicheng Lin, Junyang Lin, Xuejing Liu, Jiawei Liu, Chenglong Liu, Yang Liu, Dayiheng Liu, Shixuan Liu, Dunjie Lu, Ruilin Luo, Chenxu Lv, Rui Men, Lingchen Meng, Xuancheng Ren, Xingzhang Ren, Sibo Song, Yuchong Sun, Jun Tang, Jianhong Tu, Jianqiang Wan, Peng Wang, Pengfei Wang, Qiuyue Wang, Yuxuan Wang, Tianbao Xie, Yiheng Xu, Haiyang Xu, Jin Xu, Zhibo Yang, Mingkun

Yang, Jianxin Yang, An Yang, Bowen Yu, Fei Zhang, Hang Zhang, Xi Zhang, Bo Zheng, Humen Zhong, Jingren Zhou, Fan Zhou, Jing Zhou, Yuanzhi Zhu, and Ke Zhu. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631, 2025.

Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023.

ByteDance. Seedance 2.0, 2026. URL https://seed.bytedance.com/en/seedance2_0. Agneet Chatterjee, Rahim Entezari, Maksym Zhuravinskyi, Maksim Lapin, Reshinth Adithyan, Amit

Raj, Chitta Baral, Yezhou Yang, and Varun Jampani. Stable cinemetrics: Structured taxonomy and evaluation for professional video generation. arXiv preprint arXiv:2509.26555, 2025.

J. S. Chung and A. Zisserman. Out of time: automated lip sync in the wild. In Workshop on Multi-view Lip-reading, ACCV, 2016.

Jiankang Deng, Jia Guo, Xue Niannan, and Stefanos Zafeiriou. Arcface: Additive angular margin

loss for deep face recognition. In CVPR, 2019. Gemini Team, Google. Gemini 3 pro. 2026. URL https://gemini.google.com/app. Google Deepmind. Gemini 2.5 flash image (nano banana): Create and edit images with gemini.

https://deepmind.google/models/gemini/image, 2025a. Google Deepmind. Veo3 video model. https://deepmind.google/models/veo/, 2025b. Yuwei Guo, Ceyuan Yang, Ziyan Yang, Zhibei Ma, Zhijie Lin, Zhenheng Yang, Dahua Lin, and

Lu Jiang. Long context tuning for video generation. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 17281–17291, October 2025.

Yoav HaCohen, Nisan Chiprut, Benny Brazowski, Daniel Shalem, Dudu Moshe, Eitan Richardson, Eran Levin, Guy Shiran, Nir Zabari, Ori Gordon, Poriya Panet, Sapir Weissbuch, Victor Kulikov, Yaki Bitterman, Zeev Melumian, and Ofir Bibi. Ltx-video: Realtime video latent diffusion. arXiv preprint arXiv:2501.00103, 2024.

Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, et al. Vbench: Comprehensive benchmark suite for video generative models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21807–21818, 2024.

Ziqi Huang, Fan Zhang, Xiaojie Xu, Yinan He, Jiashuo Yu, Ziyue Dong, Qianli Ma, Nattapol Chanpaisit, Chenyang Si, Yuming Jiang, et al. Vbench++: Comprehensive and versatile benchmark suite for video generative models. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2025.

Zeyinzi Jiang, Zhen Han, Chaojie Mao, Jingfeng Zhang, Yulin Pan, and Yu Liu. Vace: All-in-one video creation and editing. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 17191–17202, 2025.

Siddharth Joshi, Haoli Yin, Rishabh Adiga, Ricardo Monti, Aldo Carranza, Alex Fang, Alvin Deng, Amro Abbas, Brett Larsen, Cody Blakeney, et al. Datbench: Discriminative, faithful, and efficient vlm evaluations. arXiv preprint arXiv:2601.02316, 2026.

Timo Kaufmann, Paul Weng, Viktor Bengs, and Eyke Hüllermeier. A survey of reinforcement learning from human feedback. arXiv preprint arXiv:2312.14925, 2023.

Rahima Khanam and Muhammad Hussain. Yolov11: An overview of the key architectural enhancements. arXiv preprint arXiv:2410.17725, 2024.

Kuaishou. Kling video model. https://kling.kuaishou.com, 2025.

Yaofang Liu, Xiaodong Cun, Xuebo Liu, Xintao Wang, Yong Zhang, Haoxin Chen, Yang Liu, Tieyong Zeng, Raymond Chan, and Ying Shan. Evalcrafter: Benchmarking and evaluating large video generation models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 22139–22149, 2024.

Luma AI. Dream machine, 2024. URL https://lumalabs.ai/dream-machine. Yihao Meng, Hao Ouyang, Yue Yu, Qiuyu Wang, Wen Wang, Ka Leong Cheng, Hanlin Wang, Yixuan

Li, Cheng Chen, Yanhong Zeng, Yujun Shen, and Huamin Qu. Holocine: Holistic generation of cinematic multi-shot long video narratives. arXiv preprint arXiv:2510.20822, 2025.

MiniMax. Hailuo ai, 2026. URL https://hailuo.ai. OpenAI. Video generation models as world simulators. https://openai.com/research/

video-generation-models-as-world-simulators, 2024. OpenAI. Sora2 video model. https://openai.com/research/sora-2, 2025. Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov,

Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023.

William S. Peebles and Saining Xie. Scalable diffusion models with transformers. Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 4172–4182, 2023. URL https://api.semanticscholar.org/CorpusID:254854389.

Qianqian Qiao, DanDan Zheng, Yihang Bo, Bao Peng, Heng Huang, Longteng Jiang, Huaye Wang, Jingdong Chen, Jun Zhou, and Xin Jin. Vadb: A large-scale video aesthetic database with professional and multi-dimensional annotations. arXiv preprint arXiv:2510.25238, 2025.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021.

Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey, and Ilya Sutskever. Robust speech recognition via large-scale weak supervision. In International conference on machine learning, pages 28492–28518. PMLR, 2023.

Haoyuan Shi, Yunxin Li, Nanhao Deng, Zhenran Xu, Xinyu Chen, Longyue Wang, Baotian Hu, and Min Zhang. Msvbench: Towards human-level evaluation of multi-shot video generation. arXiv preprint arXiv:2602.23969, 2026.

Alibaba ATH Innovation Team. Happyhorse-1.0, 2026.

Tencent, Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, Kathrina Wu, Qin Lin, Junkun Yuan, Yanxin Long, Aladdin Wang, Andong Wang, Changlin Li, Duojun Huang, Fang Yang, Hao Tan, Hongmei Wang, Jacob Song, Jiawang Bai, Jianbing Wu, Jinbao Xue, Joey Wang, Kai Wang, Mengyang Liu, Pengyu Li, Shuai Li, Weiyan Wang, Wenqing Yu, Xinchi Deng, Yang Li, Yi Chen, Yutao Cui, Yuanbo Peng, Zhentao Yu, Zhiyu He, Zhiyong Xu, Zixiang Zhou, Zunnan Xu, Yangyu Tao, Qinglin Lu, Songtao Liu, Daquan Zhou, Hongfa Wang, Yong Yang, Di Wang, Yuhong Liu, Jie Jiang, and Caesar Zhong. Hunyuanvideo: A systematic framework for large video generative models. CoRR, abs/2412.03603, 2024.

Thomas Unterthiner, Sjoerd van Steenkiste, Karol Kurach, Raphaël Marinier, Marcin Michalski, and

Sylvain Gelly. Fvd: A new metric for video generation. Openreview, 2019. Vidu. Vidu, 2024.

Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, Jianyuan Zeng, Jiayu Wang, Jingfeng Zhang, Jingren Zhou, Jinkai Wang, Jixuan Chen, Kai Zhu, Kang Zhao, Keyu Yan, Lianghua Huang, Mengyang Feng, Ningyi Zhang, Pandeng Li, Pingyu Wu, Ruihang Chu, Ruili Feng, Shiwei Zhang, Siyang Sun, Tao Fang,

Tianxing Wang, Tianyi Gui, Tingyu Weng, Tong Shen, Wei Lin, Wei Wang, Wei Wang, Wenmeng Zhou, Wente Wang, Wenting Shen, Wenyuan Yu, Xianzhong Shi, Xiaoming Huang, Xin Xu, Yan Kou, Yangyu Lv, Yifei Li, Yijing Liu, Yiming Wang, Yingya Zhang, Yitong Huang, Yong Li, You Wu, Yu Liu, Yulin Pan, Yun Zheng, Yuntao Hong, Yupeng Shi, Yutong Feng, Zeyinzi Jiang, Zhen Han, Zhi-Fan Wu, and Ziyu Liu. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025.

Qinghe Wang, Xiaoyu Shi, Baolu Li, Weikang Bian, Quande Liu, Huchuan Lu, Xintao Wang, Pengfei Wan, Kun Gai, and Xu Jia. Multishotmaster: A controllable multi-shot video generation framework. arXiv preprint arXiv:2512.03041, 2025a.

Xinran Wang, Songyu Xu, Xiangxuan Shan, Yuxuan Zhang, Muxi Diao, Xueyan Duan, Yanhua Huang, Kongming Liang, and Zhanyu Ma. Cinetechbench: A benchmark for cinematographic technique understanding and generation. arXiv preprint arXiv:2505.15145, 2025b.

Jianhui Wei, Xiaotian Zhang, Yichen Li, Yuan Wang, Yan Zhang, Ziyi Chen, Zhihang Tang, Wei Xu, and Zuozhu Liu. Univbench: Towards unified evaluation for video foundation models. arXiv preprint arXiv:2602.21835, 2026.

Weijia Wu, Zeyu Zhu, and Mike Zheng Shou. Automated movie generation via multi-agent cot planning. ArXiv, abs/2503.07314, 2025. URL https://api.semanticscholar.org/CorpusID: 276929150.

Zeyue Xue, Jie Wu, Yu Gao, Fangyuan Kong, Lingting Zhu, Mengzhao Chen, Zhiheng Liu, Wei Liu, Qiushan Guo, Weilin Huang, and Ping Luo. Dancegrpo: Unleashing grpo on visual generation,

##### 2025. URL https://arxiv.org/abs/2505.07818.

Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video diffusion models with an expert transformer. arXiv preprint arXiv:2408.06072, 2024.

Haojie Zhang, Di Wu, Bingyan Liu, Linjie Zhong, Yuancheng Wei, Xingsong Ye, Nanqing Liu, and Yaling Liang. Muss: A large-scale dataset and cinematic narrative benchmark for multi-shot subject-to-video generation. arXiv preprint arXiv:2604.23789, 2026.

Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In Proceedings of the IEEE/CVF international conference on computer vision, pages 3836–3847, 2023.

Dian Zheng, Ziqi Huang, Hongbo Liu, Kai Zou, Yinan He, Fan Zhang, Lulu Gu, Yuanhan Zhang, Jingwen He, Wei-Shi Zheng, et al. Vbench-2.0: Advancing video generation benchmark suite for intrinsic faithfulness. arXiv preprint arXiv:2503.21755, 2025.

Ziwei Zhou, Zeyuan Lai, Rui Wang, Yifan Yang, Zhen Xing, Yuqing Yang, Qi Dai, Lili Qiu, and Chong Luo. Avgen-bench: A task-driven benchmark for multi-granular evaluation of text-to-audiovideo generation, 2026.

