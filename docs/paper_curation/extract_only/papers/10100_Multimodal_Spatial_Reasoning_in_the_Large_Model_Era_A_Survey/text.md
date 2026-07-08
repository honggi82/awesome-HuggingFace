# Multimodal Spatial Reasoning in the Large Model Era: A Survey and Benchmarks

†Xu Zheng1,2, †Zihao Dongfang1, ∗Lutao Jiang1, ∗Boyuan Zheng1, ∗Yulong Guo1, Zhenquan Zhang4, Giuliano Albanese2, Runyi Yang2, Mengjiao Ma2, Zixin Zhang1, Chenfei Liao1,5, Dingcheng Zhen8, Yuanhuiyi Lyu1, Yuqian Fu2, Bin Ren6,7, Linfeng Zhang5, Danda Paudel2, Nicu Sebe7, Luc Van Gool2, ‡Xuming Hu1,3

1HKUST(GZ) 2INSAIT, Sofia University “St. Kliment Ohridski” 3HKUST 4South China University of Technology 5Shanghai Jiao Tong University 6University of Pisa 7University of Trento 8Independent † Co-first Author; ∗ Core Contributors; ‡ Corresponding Author.

## arXiv:2510.25760v2[cs.CV]2Nov2025

[Figure 1]

Fig. 1: (a) Various multimodal inputs for advanced spatial reasoning with MLLMs, such as 2D images [1], 3D scenes [2] and videos [3]. (b) Downstream tasks base or rely on spatial reasoning, such as VLA [4], 3D layout generation [5], and vision-language action [6].

I. INTRODUCTION A. Background

Abstract—Humans possess spatial reasoning abilities that enable them to understand spaces through multimodal observations, such as vision and sound. Large multimodal reasoning models extend these abilities by learning to perceive and reason, showing promising performance across diverse spatial tasks. However, systematic reviews and publicly available benchmarks for these models remain limited. In this survey, we provide a comprehensive review of multimodal spatial reasoning tasks with large models, categorizing recent progress in multimodal large language models (MLLMs) and introducing open benchmarks for evaluation. We begin by outlining general spatial reasoning, focusing on posttraining techniques, explainability, and architecture. Beyond classical 2D tasks, we examine spatial relationship reasoning, scene and layout understanding, as well as visual question answering and grounding in 3D space. We also review advances in embodied AI, including vision-language navigation and action models. Additionally, we consider emerging modalities such as audio and egocentric video, which contribute to novel spatial understanding through new sensors. We believe this survey establishes a solid foundation and offers insights into the growing field of multimodal spatial reasoning. Updated information about this survey, codes and implementation of the open benchmarks can be found at https://github.com/zhengxuJosh/Awesome-Spatial-Reasoning.

Spatial reasoning is a fundamental human ability that allows individuals to understand and interact with the world through multimodal inputs, such as vision, sound, and other senses. It supports navigation, comprehension of object relationships, and problem-solving in spatial contexts, as shown in Figure 1. While large language models (LLMs) have made significant strides in text processing and generation [55], their spatial reasoning is limited by their primarily unimodal design [56]. Integrating multimodal information—such as images, audio, and video—into language models offers new opportunities to enhance spatial reasoning, particularly for tasks requiring deep understanding of complex real-world scenarios [57–63].

Large multimodal reasoning models have emerged as a promising solution, as they are trained to perceive and reason across multiple modalities simultaneously [64–68]. These models have shown remarkable performance in a wide range of spatial tasks, from understanding 2D spatial relationships to more complex 3D reasoning. However, despite these advancements, there remains a notable gap in systematically reviewing

Index Terms—Spatial Reasoning, Multimodal Large Language Model, Survey, Benchmark

Prompt Engineering Spatial-MM [7], VSI-Bench [8], VoT [9], etc Tool Use SpatialScore [10], SpatialPIN [11], etc

Test-Time Scaling

Others VisuoThink [12], Logic-RAG [13], etc

SFT Multi-SpatialMLLM [14], SpatialVLM [15], etc

Post-Training

General MLLM

RL Video-R1 [16], Spatial-R1 [? ], etc Model Design Spatial-MLLM [17], SpatialRGPT [18], Spatial-ORMLLM [19], etc Explainability Beyond Semantics [20], ADAPTVIS [21], RelatiViT [22], etc

3D Input LLM-Grounder [23], Grounded 3D-LLM [24], etc Multi-view Input VLM-Grounder [25], 3DAxisPrompt [2], etc

3D Visual Grounding

Hybrid of 3D and 2D SeeGround [26], ReasonGrounder [27], etc

MultimodalSpatialReasoning

Training-required LLaVA-3D [28], 3DGraphLLM [29], etc Training-free SpatialPIN [11], Agent3D-Zero [30], etc

3D Vision

3D Scene Reasoning and QA

3D Layout Generation LayoutGPT [5], Layout-your-3D [31], etc 3DGen as Program 3D-GPT [32], CAD-Recode [33], etc

3D Generation

Scene Understanding Spartun3D [34], GSA-VLN [35], etc Intention Interpretation AutoSpatial [36], LL3DA [37], etc Planning & Navigation NavVLM [38], NavCoT [39], etc

Vision-Language Navigation

Embodied Question Answering OpenEQA [40], EMBOSR [41], etc

Embodied AI

Embodied Grasping ThinkGrasp [42], FreeGrasp [43], etc Vision-Language Action 3D-VLA [44], π0.5[45], Chat-VLA2 [46], etc Embodied World Model TesserAct [47], EVA [48], etc

Video-based VideoLLaMA2 [49], VideoINSTA [50], Video-R1 [16], SpaceR [3], etc Audio-based STARSS23 [51], SpatialSoundQA [52], ACORN [53], SAVVY [54], etc

Novel Modalities

Fig. 2: Taxonomy for multimodal spatial reasoning with large models.

and evaluating the performance of these emerging models, especially in the context of multimodal spatial reasoning.

- B. Contributions

This survey aims to fill that gap by providing a comprehensive review of the current state of multimodal spatial reasoning with large models, as shown in Figure 2. We begin by reviewing the general landscape of spatial reasoning, focusing on key aspects such as post-training techniques [15, 16], model explainability [20], and architecture design [18]. Moving beyond traditional 2D tasks [10], we delve into more advanced forms of spatial reasoning, including spatial relationship reasoning [39], scene and layout understanding [5], and grounding visual information in 3D space [27]. Furthermore, this paper also explores the intersection of spatial reasoning and embodied AI tasks [40], including vision-language navigation and action models [44], where models are required to perform tasks in dynamic environments based on multimodal inputs.

We extend the discussion to incorporate the use of emerging modalities such as audio and ego-centric video, which offer distinct opportunities for spatial understanding, particularly in novel sensor environments [69, 70]. In addition to reviewing the existing literature, we introduce open benchmarks for evaluating the performance of MLLMs in spatial reasoning tasks. These benchmarks aim to standardize the evaluation of these models and provide a reliable foundation for future research. The introduction of these benchmarks will also facilitate comparisons across different models and drive advancements in the field by offering standardized testing protocols.

We believe this survey serves as an essential resource for researchers and practitioners in the field of multimodal spatial reasoning, establishing a solid foundation for future work in this critical area. Additionally, we provide access to the codes, implementations, and up-to-date information about the open benchmarks at https://github.com/zhengxuJosh/ Awesome-Spatial-Reasnoning, which can help further advance

TABLE I: Recent related survey papers on Reasoning in MLLMs.

Authors Venue/Date Main Focus/Analysis Link

Zhou et al. [71] Arxiv 2025 (May) RL-based reasoning link Wang et al. [72] Arxiv 2025 (Apr) Explores small reasoning models, training, inference, and applications link Ke et al. [73] Arxiv 2025 (Apr) Discusses inference scaling, learning-to-reason, and agentic systems in LLMs link Zha et al. [56] Arxiv 2025 (Apr) Focuses on enabling LLMs with 3D spatial reasoning capabilities link Bi et al. [74] Arxiv 2025 (Apr) Reviews advancements in multimodal reasoning in LLMs link

- Chen et al. [75] Arxiv 2025 (Apr) Investigates scaling challenges and techniques in LLM reasoning link

- Chen et al. [76] Arxiv 2025 (Apr) Discusses long chain-of-thought approaches for enhancing LLM reasoning link Ali et al. [77] Arxiv 2025 (Mar) Focuses on mathematical reasoning and optimization tasks within LLMs link Wang et al. [78] Arxiv 2025 (Mar) Reviews efficient reasoning techniques for large-scale LLMs link Plaat et al. [79] Arxiv 2025 (Mar) Explores efficient inference techniques for large reasoning models link Qu et al. [80] Arxiv 2025 (Mar) Discusses language and multimodal techniques for efficient reasoning in LLMs link Lin et al. [81] Arxiv 2025 (Mar) Focuses on transitioning from language reasoning to multimodal reasoning link Sui et al. [82] Arxiv 2025 (Mar) Reviews techniques for reducing inefficiencies in LLM reasoning link Wang et al. [83] Arxiv 2025 (Mar) Examines the integration of chain-of-thought reasoning with multimodal LLMs link Bandyopadhyay et al. [84] Arxiv 2025 (Mar) Discusses various reasoning strategies implemented in LLMs link Li et al. [85] Arxiv 2025 (Mar) Focuses on methods to improve causal reasoning abilities in LLMs link Yan et al. [86] Arxiv 2025 (Feb) Reviews mathematical reasoning benchmarks and methods in LLMs link Yang et al. [87] Arxiv 2025 (Feb) Explores code-enhanced reasoning in LLMs, and reasoning-driven code tasks link Li et al. [88] Arxiv 2025 (Feb) Focuses on cognitive reasoning models and LLMs (System 1 vs System 2) link Cheng et al. [89] Arxiv 2025 (Feb) Discusses integrating logical reasoning in LLMs for more structured outputs link Srivastava et al. [90] Arxiv 2025 (Feb) Investigates small language models’ reasoning abilities and improvements link Xu et al. [91] Arxiv 2025 (Jan) Focuses on reinforced reasoning techniques for LLMs link Wang et al. [92] Arxiv 2024 (Jan) Explores the emerging trends and challenges in multimodal reasoning for LLMs link Ours Arxiv 2025 (Aug) Multimodal spatial reasoning in the large model era link

research in this domain. Through this work, we aim to provide valuable insights into the current challenges and future opportunities in multimodal spatial reasoning with large models, encouraging further exploration and development in this rapidly evolving field.

3D visual grounding), incorporate emerging modalities (audio, egocentric video), and present open benchmarks and evaluation protocols absent from prior work. This focused review aims to provide a concise foundation for advancing research and practical evaluation in multimodal spatial reasoning.

- C. Related Works

Significant progress has integrated vision, audio, and other modalities with text models, enabling richer spatial reasoning in 2D and 3D. Prior surveys examine related directions but leave gaps relevant to multimodal spatial tasks. For example, Wang et al. [72] study small reasoning models but focus on unimodal, low-complexity tasks; Ke et al. [73] analyze inference scaling and agentic systems without deeply addressing multimodal spatial reasoning; and Zha et al. [56] emphasize 3D capabilities but concentrate on implementation details rather than cross-modal evaluation. Broad reviews such as Bi et al. [74] summarize multimodal advances but do not propose systematic benchmarks or evaluation frameworks for spatial understanding in dynamic, real-world settings.

Our survey fills this gap by concentrating on multimodal spatial reasoning in the large-model era. We categorize spatial tasks (e.g., relationship reasoning, scene understanding,

II. PROBLEM SETUP: MULTIMODAL SPATIAL REASONING

Definition. Multimodal spatial reasoning aims to infer spatial relations, locations, and actions from heterogeneous inputs and to produce verifiable outputs grounded in space. Formally, given inputs X = {ximg,xvid,xpc,xaud,xtext,...} (e.g., RGB images, videos, point clouds, audio, and language) under a specified reference frame (2D/3D/ego/allo), a model predicts Y such as (i) textual answers/rationales, (ii) geometric quantities (boxes, poses, trajectories), or (iii) executable actions/plans for embodied settings. This unifies classic VQAstyle queries, 3D grounding, navigation, and layout/scene generation [18, 34, 36, 93, 94].

A. Types of Spatial Reasoning in MLLMs

Spatial reasoning in MLLMs spans basic localization to advanced scene modeling. Key types include: ① Localization &

Category Details

- 1. Localization: Locate objects in 2D/3D.
- 2. Relation: Reason about spatial relations.
- 3. Navigation: Plan paths and optimize actions.
- 4. Pattern: Detect patterns/symmetries.
- 5. Scaling: Resize while preserving proportions.
- 6. Transformation: Apply spatial changes.
- 7. Context: Interpret positions in context.
- 8. 3D Generation: Synthesize 3D scenes.
- 9. Modeling: Build scene models for predictions.
- 10. Interaction: Support real-time spatial interaction.

Types

- 1. Multimodal Integration: Test modality combinations.
- 2. Task Coverage: VQA, 3D localization, navigation.
- 3. Transparency: Trace decisions with maps or probes.
- 4. Generalization: Test adaptability in new environments.
- 5. Embodied Testing: Measure real-time performance.
- 6. Benchmarking: Provide reproducible tasks.

Eval

- 1. 2D Tasks: Spatial reasoning in images/videos.
- 2. 3D Reasoning: Grounding, QA, navigation.
- 3. Embodied Reasoning: Navigation and world models.
- 4. Novel Modalities: Cross-domain spatial reasoning.

Roadmap

- TABLE II: Overview of Spatial Reasoning in MLLMs: Types, Evaluation Protocols, and Roadmap

Memory: Locate objects in 2D/3D relative to others/observer and track their states over time. ② Relation & Geometry: Reason about spatial relations (above/below/left/right) and metrics (distance, angle, area, volume). ③ Navigation & Problem Solving: Plan paths and optimize actions (e.g., shortest routes, spatial puzzles). ④ Pattern & Perspective: Detect patterns/symmetries and reason across viewpoints. ⑤ Scaling & Resizing: Model size changes while preserving proportions. ⑥ Transformation: Apply rotation, translation, and scaling while maintaining relationships. ⑦ Contextualization: Interpret positions under environmental context (e.g., room vs. spacecraft). ⑧ 3D Model Generation: Synthesize 3D shapes/scenes from spatial cues. ⑨ Environmental Modeling: Build scene/world models for prediction and decision making. ⑩ Sensing & Interaction: Support real-time spatial interaction (e.g., AR) via sensors/vision. These abilities underpin applications from navigation to simulation and interactive systems.

- B. Evaluation Protocols for Spatial Reasoning

Evaluating MLLMs’ spatial reasoning should probe accuracy, robustness, interpretability, and generalization. Key dimensions: ① Multimodal Integration: Test diverse modality combos (images, text, audio, depth/point clouds, sensors) to assess cross-modal fusion beyond unimodal cues. ② Task Coverage: Include VQA, 3D localization, map-based navigation, embodied planning, and scene/image generation to span low- and high-level reasoning. ③ Process Transparency: Trace decisions via attention maps, intermediate states, or rationale probes to reveal how spatial relations are encoded/manipulated. ④ Generalization & Robustness: Evaluate out-of-distribution settings (novel layouts, unseen environments, perturbations) to test adaptability. ⑤ Interactive/Embodied Testing: Measure real-time performance for navigation/manipulation and AR/VR, including responsiveness and online updates. ⑥ Benchmark Standardization: Provide

Test-Time Scaling Post Training Explainability

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

Large Language Models

Connector

Modality Encoder

Tokenizer

[Figure 6]

…

Image Video 3D

Text

Audio

Fig. 3: Typical MLLM architecture and strategies.

reproducible suites spanning controlled synthetic tasks and real-world scenarios. Addressing these facets enables comprehensive, comparable assessment of MLLMs’ spatial reasoning and clarifies strengths/weaknesses across applications.

Roadmap. We next instantiate this setup across application strata: (1) general 2D image/video tasks with MLLMs, (2) 3D spatial reasoning (grounding, QA, navigation), and (2) embodied spatial reasoning (VLN, VLA, world model), and (3) novel modalities & cross-domain settings. Each section maps back to the taxonomy above and adopts the evaluation dimensions outlined here.

III. GENERAL MULTIMODAL SPATIAL REASONING

General multimodal spatial reasoning refers to MLLMs’ ability to understand and reason about spatial relationships across visual and textual inputs. It encompasses tasks such as visual question answering (VQA) on spatial relations, object localization, perspective understanding, 3D comprehension, and navigation. These tasks require aligning visual perception with linguistic expressions of spatial concepts like “above,” “behind,” and “to the left of.” As shown in Figure 3, current research enhances spatial reasoning in multimodal models along four main directions: ① Test-time scaling to boost inferencetime capability; ② Post-training methods such as supervised fine-tuning and reinforcement learning on spatial datasets; ③ Architectural improvements for richer spatial encoding; and ④ Explainability studies to reveal limitations and failure modes in spatial reasoning.

A. Test-Time Scaling Methods

Test-time scaling methods offer training-free strategies to enhance MLLMs’ spatial reasoning during inference. Instead of retraining or fine-tuning, these approaches leverage improved prompting, tool-assisted reasoning, and external modality integration. Existing works can be broadly grouped into three categories based on their methodological focus.

- TABLE III: Comparison of prompt engineering methods for multimodal spatial reasoning. We summarize key ideas and prompt types of representative approaches.

Method Prompt Type Key Idea / Mechanism TopViewRS [95] Textual Uses simple top-view templates as

baseline prompts for spatial reasoning.

VSI-Bench [8] Textual (graph-structured) Guides models to build and use cognitive graphs for spatial distance reasoning.

OmniSpatial [96] Textual (CoT) Applies Chain-of-Thought reason-

ing to spatial VQA tasks.

SCABenchmark [97] Textual (structured cues) Adds coordinates and reference frames to improve spatial understanding.

Spatial-MM [7] Visual Uses bounding boxes or scene graphs to enhance spatial reasoning accuracy.

Mind’s Eye (VoT) [9] Visual / Hybrid Visualizes reasoning steps as spa-

tial traces to aid understanding.

SpatialPIN [11] Progressive Decomposes complex spatial

queries into multi-stage sub-tasks.

SpatialPrompt [98] Quantitative / Textual Establishes spatial anchors for

stepwise geometric reasoning.

SpatialMind [99] Structured / Multi-modal Integrates scene representations with task-specific reasoning plans.

1) Prompt Engineering: Prompt engineering is the most direct and lightweight approach to enhance spatial reasoning in MLLMs without external tools or fine-tuning. Recent work explores how carefully crafted prompts can better elicit models’ latent spatial reasoning abilities. Although Chain-of-Thought (CoT) prompting has achieved notable success in general reasoning, its direct application to spatial tasks yields limited gains. To address this, researchers have proposed specialized prompting strategies tailored for spatial understanding, as shown in Table III.

Early methods, such as TopViewRS [95], introduce simple templates but show only marginal improvements. VSIBench [8] demonstrates that explicitly instructing MLLMs to build cognitive graphs enhances spatial question answering, whereas standard CoT fails. Similarly, OmniSpatial [96] finds textual CoT ineffective for complex perspectivetaking. SCABenchmark [97] further analyzes prompt formats and frames of reference, showing that explicit geometric and relational cues—like coordinates and reference frames—outperform long, free-form CoT reasoning. Beyond text, visual prompting has proven complementary. Spatial-MM [7] shows that supplying bounding boxes or scene graphs—either annotated or self-generated—greatly improves multi-hop spatial reasoning, where CoT alone fails. Mind’s Eye [9] extends this with the Visualization-of-Thought paradigm, where the model visualizes reasoning traces during inference, significantly boosting 2D spatial reasoning accuracy.

Additionally, progressive prompting frameworks decompose complex queries into manageable steps. SpatialPIN [11] employs multi-stage prompting with dense visual priors from multiple vision foundation models, demonstrating the benefits of structured, incremental reasoning. For quantitative spatial

TABLE IV: Summary of tool-usage methods for multimodal spatial reasoning. ✓ indicates the method supports the feature.

Appendimages/traces

Serializetokens/BEV

Rendernovelviews

UIops(crop/seg)

Agenticcontrol

2Dperception

Plan–Execute

3Drecon

ReAct

Method

IoT [100] ✓ ✗ ✗ ✓ ✗ ✗ ✗ ✗ ✗ Struct2D [101] ✗ ✓ ✗ ✗ ✓ ✗ ✗ ✗ ✗ Lee et al. [102] ✗ ✓ ✗ ✗ ✓ ✗ ✗ ✗ ✗ ZeroVLM [103, 104] ✗ ✗ ✓ ✓ ✗ ✓ ✗ ✗ ✗ SpatialPIN [11] ✗ ✓ ✓ ✗ ✓ ✓ ✗ ✗ ✗ VADAR [105] ✗ ✓ ✗ ✗ ✓ ✗ ✓ ✓ ✗ SpatialAgent [10] ✗ ✓ ✗ ✗ ✓ ✗ ✓ ✓ ✓

reasoning, SpatialPrompt [98] improves performance by establishing explicit spatial anchors and prompting stepwise transformations relative to them. SpatialMind [99] integrates scene representations—modeled as object-centric text, 2D grids, or 3D maps—with question-type–specific reasoning plans (e.g., locate → transform → compare), guiding more systematic inference at test time.

Insights & Discussion. The evolution from simple CoT prompting to spatially structured prompting reveals a key distinction between linguistic and spatial reasoning in MLLMs. While textual CoT assumes that verbalizing intermediate steps improves reasoning, spatial reasoning requires explicit modeling of visual relations—through visual traces, structured graphs, or reference-based transformations. This indicates that effective spatial prompting depends less on longer reasoning chains and more on aligning prompt representations with the inherently visual and relational nature of spatial cognition. Future work may explore adaptive prompting frameworks that automatically select the most suitable representational format—textual, visual, or hybrid—based on the type of spatial query and reasoning context.

2) Tool Usage: Integrating tools at test time enhances MLLMs’ spatial reasoning by providing explicit geometric or structural priors without modifying the base model. As in Table IV, three main tool families have emerged. First, UI-style visual operations (e.g., crop, zoom, mark, edge, and segmentation) expose fine-grained spatial cues often missed by MLLMs. For instance, Image-of-Thought [100] directs the model to plan and execute short visual operation sequences, generating “visual rationales” that are fed back alongside text reasoning. Second, 2D perception modules—such as object detection, orientation, depth, and pose estimation—convert pixels into structured, object-centric facts. Struct2D [101] renders a BEV canvas with filtered object marks and metadata (IDs, categories, coordinates), while Lee et al. [102] constructed abstract scene layouts to support perspective transformations. Third, 3D reconstruction tools lift images into view-consistent geometry for perspective-based reasoning. ZeroVLM [103] employs Zero-1-to-3 [104] to synthesize novel views and pair them with “view prompts” that anchor camera relations, and

SpatialPIN [11] partially reconstructs lightweight 3D objects for downstream spatial queries.

Given these tools, inference-time integration generally follows three escalating patterns. First, some methods append tool-generated images or traces to the input: IoT concatenates cropped or segmented snippets as visual evidence, while ZeroVLM stitches multi-view mosaics with view-aware prompts [100, 103]. Second, others serialize perception into structured tokens or sketches: Struct2D supplies a BEV bitmap with concise object metadata, and Lee et al. inject numeric orientations and perspective descriptors to convert allocentric queries into egocentric ones [101, 102]. Finally, 3D-aware approaches render novel views from reconstructed geometry: ZeroVLM generates left/right/random perspectives to test viewpoint sensitivity, while SpatialPIN’s partial 3D lifting enables virtual viewpoints that re-ground spatial relations [11, 103].

Beyond single-shot prompting, modern systems increasingly control tools through agentic policies at inference. VADAR (Visual Agentic AI) [105] designs a dynamic API and synthesizes short programs that call specialized modules (detector, depth, pose) on demand—illustrating “plan-to-execute” tool use via code generation for reliable multi-step reasoning. SpatialScore’s SpatialAgent [10] provides a standardized multiagent framework with nine spatial tools and two control paradigms: a hierarchical Plan–Execute pipeline and an interleaved ReAct mode that alternates reasoning and action, enabling consistent cross-method evaluation. Diagnostics further reveal which tool outputs matter most. Ravi et al. [106] show in Disjoint-3DQA that trajectories or BEV features offer limited gains across non-co-visible frames, while oracle 3D coordinates yield substantial improvements—highlighting metrically faithful 3D states or persistent scene memory as the most effective feedback signals.

Insights & Discussion. Test-time tool use works by externalizing geometry into inputs MLLMs already consume—visual traces, structured tokens, and novel views—rather than elongating textual CoT. Gains are largest when signals are metrically grounded (poses, coordinates, calibrated depth) and agentic controllers compose tools into reusable subroutines, improving perspective shifts, occlusions, and multi-object relations without retraining. ① Remaining issues: perception and view-synthesis errors propagate without uncertainty handling;

- 2D proxies (BEV, trajectories) poorly approximate metric
- 3D state; temporal persistence is weak—no durable, objectcentric world memory; and tool outputs lack standardized units/frames, harming alignment and reproducibility. Multitool pipelines also add cost and latency for open-world, longhorizon tasks. ② Promising directions: maintain a persistent object-centric scene memory with cross-view/time checks and lightweight geometric self-verification; standardize tool outputs (schemas for objects/cameras/constraints with calibrated uncertainty) to enable evidence weighting and conflict resolution; and develop budget-aware controllers that switch between Plan–Execute and ReAct, add verify–reflect loops, and distill heavy chains into compact prompts/plugins—evaluated with utility–cost–robustness metrics in long-horizon, non-covisible, open-world regimes.

3) Others: Beyond prompt engineering and tool use, several training-free inference strategies improve spatial reasoning. The first category is self-consistency voting. Sample multiple reasoning chains and take a consensus to stabilize answers under perspective shifts and multi-object relations. Secondly, multimodal search explores and prunes visualspatial reasoning paths at test time; e.g., VISUOTHINK performs look-ahead tree search over interleaved visual-textual steps and selects the best-scoring solution under spatial constraints [12]. There are also retrieval-augmented generation (RAG) methods. Inject external spatial knowledge at inference. LOGIC-RAG [13] builds a dynamic first-order logic knowledge base (object positions/relations) from visual input and feeds these facts to the model, increasing driving-scene spatial accuracy from ∼55–75% to >80–90%. Grounding in retrieved maps/KBs or computed facts reduces hallucinations and sharpens spatial relations.

Insights & Discussion. Enhancing spatial reasoning in MLLMs often requires more than static prompts or single-pass outputs. Exploring multiple reasoning paths, retrieving external spatial knowledge, performing light test-time adaptation, and preserving spatial context collectively scale inferencetime capability and complement prompt/tool methods. These approaches carry trade-offs—e.g., multi-sampling and adaptation increase compute, while retrieval depends on knowledge quality—but they point toward MLLMs that dynamically and reliably reason about space with higher accuracy.

B. Post-Training Methods

Post-Training methods enhance spatial reasoning by adapting MLLMs after pre-training, mainly through supervised finetuning and reinforcement learning (RL). These approaches rely on spatially targeted datasets, rewards, and curricula to strengthen model understanding of geometry and motion.

1) Supervised Fine-tuning (SFT): SFT advances spatial reasoning by progressively broadening supervision from domainspecific static scenes to dynamic, temporally grounded reasoning. On the data side, domain-grounded QA continues to seed robust priors. CITYGPT [107] injects urban navigation and landmark knowledge through structured instructions, while MULTI-SPATIALMLLM [14] moves from single images to multi-frame settings, annotating frame-level relations (e.g., depth, camera/object motion) to capture persistence and occlusion. Extending this trend, LLAVA-ST [108] aligns finegrained spatio-temporal understanding by coupling language with explicit coordinates and temporal anchors, and STTHINK [109] focuses the lens on egocentric 4D reasoning to expose viewpoint changes and long-horizon temporal cues missing from static corpora. Synthetic pipelines complement real data: SAT [110] generates interactive, motion-centric tasks in simulation to cover self-motion and object-motion factors, and SPARE [111] automatically distills spatial QA from long-form descriptions to relieve the long-tail sparsity of rare relations. In between these regimes, SPATIALVLM [15] augments instruction tuning with region tags and relativeposition tokens (left-of, in-front-of, between, etc), pairing layout-driven QA and referring expressions so that textual

TABLE V: Comparison of reinforcement learning methods for spatial reasoning in MLLMs. ✓ indicates the presence of a feature, × indicates absence.

Self-Play/Exploration

TemporalConsistency

Process-levelReward

CurriculumLearning

3DSpatialMetrics

RewardDesign

Method

Video-R1 [16] ✓ ✗ ✗ ✗ ✗ ✓ Spatial-R1 [3] ✓ ✓ ✗ ✗ ✗ ✓ MetaSpatial [116] ✓ ✓ ✓ ✗ ✓ ✗ R1-Zero [117] ✓ ✗ ✗ ✓ ✗ ✗ ST-Think [109] ✓ ✗ ✗ ✗ ✗ ✓ M2-Reasoning [118] ✓ ✗ ✓ ✗ ✗ ✓

predicates are explicitly bound to coordinates/regions rather than inferred implicitly.

Training strategy then ties these sources together. Curricula that progress from perception to composition remain effective: SPARKLE [112] stages supervision from detection/localization toward multi-hop spatial reasoning. In parallel, motion-aware instruction tuning such as ST-VLM [113] makes kinematics explicit with trajectory-style hints. Multi-stage alignment further stabilizes learning: LLAVA-ST [108] couples semanticto-coordinate alignment with video-aware objectives, whereas SAT [110] interleaves dynamic spatial tasks as higher-level “sub-curricula” to encourage transfer from static to viewpointshifting scenarios. Looking beyond plain instruction tuning, “thinking” overlays also matter: VISUALIZATION-OFTHOUGHT [114] and VISUAL+TEXTUAL THINKING [115] introduce multimodal reasoning traces (textual steps with region/coordinate cues), nudging the model to externalize intermediate spatial inferences rather than collapsing them into a single answer token.

Insights & Discussion. SFT highlights the value of taskspecific data and structured curricula for strengthening spatial reasoning in MLLMs. Compared with pre-training alone, spatially grounded supervision enables models to internalize explicit spatial relations, motion cues, and temporal dependencies often missing in general multimodal data. Methodologically, SFT studies show that gradual exposure to increasing spatial complexity—starting from low-level perception (e.g., object localization) to higher-order reasoning (e.g., trajectory prediction, multi-hop inference)—consistently improves model performance. Incorporating temporally annotated or motionaware datasets further allows models to reason over both static configurations and dynamic evolution. Nonetheless, current SFT methods depend heavily on human-labeled or synthetic data, limiting scale and diversity. Future work could focus on automatically generating spatial annotations, leveraging selfsupervised pretexts, or designing adaptive multi-task curricula that balance static and dynamic reasoning. Ultimately, effective SFT should align supervision with the cognitive structure of spatial reasoning, bridging perception and high-level spatial understanding.

2) Reinforcement Learning (RL): RL enhances spatial reasoning by optimizing models through reward-driven feedback rather than explicit supervision.

On rewards, as in Table V VIDEO-R1 [16] introduces a time-order–aware signal (e.g., preferring correct answers on ordered vs. shuffled clips) to explicitly reward temporal use, while SPATIAL-R1/SPACER [3] extends beyond outcome rewards to process-aware credit for intermediate steps (e.g., partial route/landmark correctness, local relation checks) to improve reward stability. For 3D layout and interaction, METASPATIAL [116] blends format checks, physical feasibility, and rendering-based validation—together with object-level modulation—for consistent spatial plans. Unifying general and spatial reasoning, M2-REASONING [118] adopts task-specific RLVR signals (e.g., coordinate/ordering correctness) so that spatial subtasks contribute targeted feedback without derailing broader multimodal skills.

Training strategies typically follow a staged recipe—warm up with SFT, then refine with RL, and finally stabilize with self-improvement. VIDEO-R1 [16] uses SFT to initialize video reasoning and then applies temporally sensitive RL to consolidate it. Similarly, ST-THINK [109] employs Long-CoT SFT followed by GRPO; meanwhile, reverse thinking is used as the explicit thought style in RL, strengthening bidirectional spatial recall. METASPATIAL [116] employs curriculum-style increases in scene difficulty and multi-round refinement so that rewards stay informative as tasks grow more complex. Selfplay closes the loop: R1-Zero–like training [117] generates and solves spatial puzzles autonomously, reducing dependence on human labels and converting search over solutions into search over training data. In broader multi-task settings, M2REASONING [118] interleaves spatial RLVR with generalpurpose tasks and dynamic scheduling, mitigating interference while retaining cross-task transfer.

Overall, these approaches illustrate how RL advances spatial reasoning from two complementary angles: (1) reward design, which explicitly encodes geometric and temporal correctness; and (2) self-improvement, where models iteratively refine reasoning through autonomous exploration. Compared with supervised fine-tuning, RL offers a more flexible framework for post-training adaptation—enhancing spatial consistency, dynamic reasoning, and generalization without modifying the base architecture.

Insights & Discussion. Reinforcement learning (RL) provides a powerful framework for improving spatial reasoning in MLLMs by optimizing beyond static supervision. The reviewed methods reveal a clear evolution: from composite tasklevel rewards (VIDEO-R1) to process-level and curriculumbased optimization (SPATIAL-R1, METASPATIAL), and finally to autonomous self-play learning (R1-ZERO). This progression reflects a shift from externally guided training toward self-improving spatial cognition.

Two primary insights emerge. First, reward granularity matters—integrating intermediate reasoning rewards and geometric correctness encourages stable and interpretable spatial learning. Second, autonomous exploration enables continual improvement without reliance on labeled data, a promising direction for scalable spatial intelligence.

However, current RL frameworks remain constrained by high computational cost, reward sparsity, and limited generalization across 2D–3D–temporal domains. Future research could develop hybrid paradigms that combine RL with supervised fine-tuning or self-distillation, using automatically generated spatial feedback signals. Advancing toward richer, selfsupervised spatial rewards and cross-domain generalization will be key to achieving more human-like spatial reasoning in multimodal large language models.

- C. MLLM Architectural Modifications

Beyond post-training, architectural changes are essential for enabling MLLMs to reason about space effectively. Most MLLMs adopt a standard three-part structure—a pre-trained LLM, a visual encoder, and a modality alignment interface [64, 119–123]. However, spatial reasoning demands explicit preservation of positional and geometric information, which these components alone cannot ensure. Recent studies have thus proposed modifications to inject spatial knowledge either at the input level or via specialized model components.

1) Enhancing Input Representations: One strategy is to augment the model inputs with additional spatial cues so that the LLM can infer geometric relations without changing the core architecture.

The most straightforward one, SPATIALLLM [124], adopts a composite 3D information design, where the vision front-end mixes features from a language-supervised encoder (CLIP) with features from a self-supervised encoder (DINOv2 or MAE) to improve the 3D perception capability at the input level. Going further, MPDRIVE [125] adds an extra “marker” channel to each video frame, overlaying simple glyphs or numeric labels at detected object centers. The model processes the original RGB frame and this marker map in parallel (dual-stream), effectively bridging visual coordinates with language; this yields improved spatial understanding on autonomous driving VQA tasks. Similarly, LOCVLM [126] appends normalized (x,y) location coordinates of salient objects directly into the text prompt (treating location as part of the language input). By doing so, the LLM is encouraged to reason about spatial relations (e.g., “left of”, “inside of”) using these coordinate tokens, all without altering the pretrained vision encoder or adding new visual branches. Both methods inject explicit spatial information into the model’s context, which in turn guides the language model to produce spatially-aware descriptions and answers. Another direction is to incorporate depth and 3D cues as part of the input. SPATIALBOT [127] feeds the model with both an RGB image and its corresponding depth image (e.g., from a monocular depth estimator), essentially giving the MLLM a pseudo-3D view of the scene. This simple input-level fusion of color and depth significantly boosts the model’s depth perception and spatial QA performance, as evidenced by improvements on the SpatialQA benchmark and embodied AI tasks. Rather than images, SSR [128] leverages depth information in textual form: it converts raw depth maps into structured natural-language rationales describing the 3D layout (e.g., relative distances, sizes, and occlusions). These intermediate text descriptions are

provided to the LLM (as a chain-of-thought prompt) to guide its reasoning and are later distilled into latent embeddings for efficiency. This rationale-guided approach enables the model to utilize depth cues for higher-order spatial reasoning without requiring special sensors at inference. In a similar vein, other works enrich the model’s visual context by supplying multiple views or an explicit 3D scene representation. For instance, the SPATIO-TEMPORAL LLM framework [129] can input an entire point cloud of the environment alongside an egocentric video clip, allowing the LLM to consider the global 3D scene while also tracking temporal events. Experiments show that feeding both the holistic point cloud and video frames (plus text) enables better spatial understanding of environments and improves temporal grounding of actions. Likewise, MMSPATIAL [130] explores training MLLMs with multi-view images of a scene and their associated metric depth values. By exposing the model to multiple perspectives and precise depth measurements during fine-tuning (via the CA-VQA dataset), MM-Spatial achieves state-of-the-art 3D spatial understanding; notably, it can estimate object sizes and distances with accuracy on par with dedicated monocular depth estimators. In summary, these input-centric approaches enhance spatial reasoning by explicitly encoding geometry into the model’s inputs (either as augmented images or as location/depth tokens in text). This mitigates the loss of spatial information in standard vision backbones and provides the LLM with a richer basis for spatial inference.

Insights & Discussion. Input-centric augmentation remains minimally invasive: marker channels or coordinate tokens guide the LLM toward geometry without altering backbones, while depth, multi-view, or point-cloud evidence supplies 3D context that strengthens grounding. Yet performance is tightly coupled to detector/depth fidelity, and longer contexts strain alignment and attention memory. Uncertainty-aware spatial tokenizers and differentiable 2D–3D projectors that compress geometry, paired with curricula that progress from single-view to spatio-temporal inputs, are likely to curb shortcut reliance and improve cross-domain generalization.

2) Redesigning Spatial Reasoning Modules: An alternative (and complementary) approach introduces dedicated architectural modules that are tailored for spatial and relational reasoning. Here, the base MLLM architecture is extended with new components (or entire sub-networks) that preserve spatial structure through the model’s internal representations. For example, SPATIAL-MLLM [17] introduces a dedicated spatial encoder built on a lightweight VGGT backbone. Given sampled video frames, this encoder produces 3D-aware features that retain scene geometry. These features are then linearly projected to match both the dimensionality and the effective batch size of features from a conventional 2D visual encoder. The two streams are concatenated and passed through a modality bridge—a lightweight MLP—that converts them into unified visual tokens, which are consumed alongside text tokens by a shared LLM backbone. This geometrypreserving, spatio-temporal pathway yields consistent gains on spatial benchmarks, reporting 35-45% relative improvements over strong baselines. Similarly, SPATIAL-ORMLLM [19] incorporates a Spatial-Enhanced Feature Fusion block within

the vision tower to inject 3D understanding. In this design, 2D image features are combined with rich 3D cues (e.g., depth or volumetric estimates obtained via an external algorithm) inside a fusion module, and the resulting 2D+3D feature is fed into the LLM’s visual encoder. This end-to-end architecture effectively endows the model with volumetric spatial reasoning using only monocular RGB input, achieving robust 3D scene understanding in complex environments (like surgical operating rooms) without additional sensors. Another notable system, SPATIALRGPT [18], integrates spatial reasoning capabilities by adding a plug-in depth module and leveraging region-level training signals. In particular, SpatialRGPT uses a “flexible” depth-integration module that attaches to the existing visual encoder, enabling it to process inferred depth maps alongside RGB features. Moreover, it is trained with a curated pipeline of 3D scene-graph data to learn detailed regional representations, which allows the model to interpret user-provided region proposals and accurately judge their relative directions and distances during inference. This yields marked improvements in spatial question-answering, both with and without explicit region prompts. Yet another architectural innovation is found in CAMBRIAN-1 [131], a vision-centric multimodal model that introduces a Spatial Vision Aggregator (SVA). The SVA is a dynamic, spatially-aware connector module that fuses highresolution visual feature maps into the LLM while intelligently reducing the number of visual tokens required. By preserving fine-grained spatial information from the vision encoder and feeding it more efficiently to the language model, Cambrian-

- 1 achieves better visual grounding and overall multimodal performance (it served as an open-source testbed that reached state-of-the-art results on a new CV-Bench benchmark).

Across these designs, the common theme is the addition of structural bias for space: by introducing new layers or networks devoted to geometric processing (be it via explicit spatial feature fusion, graph relationships, or high-res feature aggregation), the models can maintain spatial layouts through the reasoning process, instead of relying solely on implicit signals in the image embeddings.

Insights & Discussion. Dedicated modules inject geometric inductive bias: multi-scale encoders, relation graphs, and spatial cross-attention preserve layout/topology; domain-tailored

- 2D+3D fusion and depth-integrated connectors enhance robustness under occlusion and clutter. Furthermore, visioncentric aggregators retain fine spatial detail with fewer tokens, and aligning static 3D context with video stabilizes temporal grounding. Nevertheless, added complexity, latency, and reliance on pseudo-3D labels motivate intent-aware routing between spatial modules and the LLM, unified 2D/3D/temporal consistency objectives, and lightweight hardware-friendly spatial layers for deployment.

- D. Explainability of Multimodal Spatial Reasoning

Understanding why MLLMs struggle with spatial reasoning is essential for advancing their design and interpretability. Recent studies have provided valuable insights into these limitations and suggested strategies for improvement.

From a mechanistic perspective, Rajabi et al. [132] reveal through attention visualization that current MLLMs often

rely on object co-occurrence rather than genuine geometric grounding. To address this, they propose decomposing spatial descriptions into grounded subject–object–relation triplets, linking detection and positional features through a lightweight relational bridge.

Following this thread, Qi et al. [20] identify a representational imbalance in multimodal Transformers where dominant vision embeddings suppress positional encodings, erasing spatial order. Using interpretability metrics, they attribute this to cross-modal norm disparities and propose normalizing vision token magnitudes and injecting mid-layer geometric features to recover spatial sensitivity without altering the backbone.

Chen et al. [21] further analyze attention maps and found that only 15–20% of attention weights target regions encoding spatial relationships, indicating that MLLMs focus on isolated objects instead of inter-object relations. They propose ADAPTVIS, a training-free inference strategy that dynamically adjusts attention based on confidence, helping the model refocus on relevant spatial regions. This process-level modulation highlights attention control as an effective route to better spatial grounding.

In parallel, Wen et al. [22] show that even large MLLMs often depend on bounding-box heuristics instead of genuine relational cues. They recast spatial relation prediction as a global object–object interaction problem and introduce RelatiViT, a transformer that integrates relation-awareness directly into self-attention, embedding structural bias for spatial reasoning into the encoder itself.

Finally, Zhang et al. [133] take a broader view, showing that simply scaling multimodal data yields diminishing gains on spatial reasoning tasks. Their analysis indicates that spatial competence relies more on the positional fidelity of the vision encoder than on the LLM’s textual positional signals. They advocate embedding explicit 3D-aware modules and crossview fusion layers to ensure spatial understanding emerges from structure rather than scale.

Insights & Discussion. Together, these studies converge on a shared diagnosis: MLLMs exhibit strong semantic reasoning but weak spatial grounding due to representational imbalance, attention bias, and lack of geometric priors, which emphasize the need for models that balance semantic and spatial representations. Future research should focus on integrating these complementary insights—explicit spatial grounding, balanced cross-modal encoding, relation-aware attention, and geometryinformed architectural priors—to enhance the accuracy and robustness of MLLMs in reasoning about spatial configurations.

IV. MULTIMODAL SPATIAL REASONING IN 3D SPACE

Multimodal spatial reasoning in 3D space is a key area of research, with significant implications for downstream applications such as navigation [38, 39], vision-language-action tasks [139, 140], and more. This section focuses on foundational tasks with multimodal spatial reasoning, including 3D grounding, 3D scene reasoning, and 3D generation. As illustrated in Figure 4, we provide an overview of these core tasks, highlighting their roles within the broader landscape of 3D spatial understanding.

Year Method Input Backbone Highlights

- 2023 LLM-Grounder [23] Point Cloud GPT-4 Uses LLM as an agent for 3D closed-loop, feedback-driven visual grounding,

which is fully zero-shot and open-vocabulary

- 2023 Grounded 3D-LLM [24] Point Cloud Tiny-Vicuna-1B Unifies 3D task modeling with LLM

- 2024 Vigor [134] Point Cloud GPT-3.5-Turbo Introduces referential order modeling for language-obj structure

- 2023 ViewRefer [135] Multi-View Image GPT-3 Multi-view modeling improves spatial perception

- 2023 3DAxiesPrompts [2] Multi-View Image GPT-4V First to encode 3D coordinates into prompt input

- 2024 VLM-Grounder [25] Multi-View Image GPT-4V Utilizes dynamic stitching strategy that dynamically uses the optimal layouts

to stitch images, enhancing VLM’s performance

- 2024 SpatialRGPT [18] RGB-D LLaMA2-7B Modular design enables flexible integration

- 2024 ZSVG3D [136] RGB-D GPT3.5 First use of program generation in 3DVG

- 2024 SeeGround [26] Text+RGB+3D Qwen2-VL-72B Dynamically adjusts perspectives to capture essential details

- 2025 ReasonGrounder [27] RGB+3DGS LLaVA 1.5 Integrates LVLM, 3DGS, and hierarchical features, enables amodal perception

under occlusion

TABLE VI: Comparison of recent multimodal spatial reasoning methods in 3D Grounding.

[Figure 7]

Fig. 4: An overview of core spatial reasoning tasks in 3D space, including 3D visual grounding[18, 134], 3D scene reasoning[11, 37], and 3D generation[137, 138].

[Figure 8]

Fig. 5: 3D visual grounding with MLLM [23].

A. 3D Visual Grounding

As in Figure 5, given a natural language description, 3D grounding involves localizing an object in a 3D scene. This task requires strong spatial reasoning to handle complex instructions and is crucial for robotics and AR, combining language understanding and 3D spatial reasoning. Traditional 3D

grounding methods are fully supervised on limited 3D datasets with predefined object captions [141], but they struggle to generalize to unseen objects and handle complex texts.

Unlike traditional methods, researchers are developing approaches based on MLLMs, significantly enhancing generalizability by leveraging large-scale priors. However, integrating MLLMs into 3D grounding remains challenging [142]. Existing approaches for embedding MLLMs into 3D grounding systems can be broadly categorized based on the input data modality: ① direct utilization of 3D representations and spatial information; ② generation of multi-view 2D images rendered from 3D scenes; ③ hybrid methods combining both 2D and 3D modalities, as shown in Table VI.

1) 3D Input: Some methods perform spatial reasoning by embedding 3D formats—such as point clouds, voxels, or learned volumetric features—into MLLMs [23, 24, 134]. LLM-Grounder [23] adopts a coarse-to-fine approach, first using an MLLM to parse complex linguistic concepts and an open-vocabulary 3D vision module to generate candidate proposals, then evaluating their semantic alignment with the query. Grounded 3D-LLM [24] integrates scene-referent to-

kens into the MLLM and employs alignment training to enable 3D input, leveraging the MLLM’s reasoning capabilities. Vigor [134] focuses on interpreting spatial language by using an LLM to infer the referential order of entities, enhancing fine-grained spatial reasoning.

Insights & Discussion. In summary, these approaches focus on 3D visual grounding by embedding 3D representations into MLLMs and utilizing their spatial reasoning ability. However, while embedding 3D modalities holds great potential, it presents challenges. The complexity of 3D data structures can hinder model interpretability, and the limited availability of labeled 3D datasets constrains the development of robust, generalizable models for open-world applications.

2) Multi-view Input: While 3D point clouds provide explicit scene representation, they present challenges for models due to the complexity of spatial information. To address this, researchers are increasingly adopting multi-view 2D representations as a promising alternative. This approach leverages the spatial reasoning capabilities of existing 2D MLLMs with minimal modifications. Representative methods include ViewRefer [135], VLM-Grounder [25], and 3DAxisPrompt [2].

A key challenge in multi-view 3D visual grounding is view discrepancy, which arises from the misalignment between the model’s perspective and the source of the grounding instruction. Several methods have been proposed to mitigate this issue. For example, ViewRefer [135] introduces learnable multi-view prototypes to capture inter-view relationships and enable knowledge transfer. VLM-Grounder [25] dynamically stitches image sequences and incorporates a grounding-andfeedback mechanism. 3DAxisPrompt [2] enhances the realworld scene by inserting 3D coordinate axes.

Insights & Discussion. These works leverage powerful MLLMs to align with 3D scenes using 2D multi-view inputs. However, key challenges remain [18]: First, MLLMs designed for global image understanding struggle with parsing specific object regions. Second, spatial perception extends beyond RGB data and requires geometric information like depth or spatial coordinates.

3) Hybrid of 2D and 3D: To combine the advantages of both 3D and multi-view representations, recent methods utilize hybrid inputs, including [18, 26, 136, 143]. SpatialRGPT [18] highlights the limitations of MLLMs relying solely on RGB pixels for 3D tasks. It proposes integrating relative depth maps from depth prediction models with RGB images to enhance spatial perception and reasoning. ZSVG3D [136] defines a visual program interface to standardize spatial relationships, enabling reasoning plans for grounding. SeeGround [26] integrates 2D visuals with explicit 3D spatial descriptions to improve object localization. 3D-MOOD [144] achieves monocular open-set 3D object detection via lifting the open-set

- 2D detection into 3D space. ReasonGrounder [27] introduces
- 3D Gaussian splatting features as intermediate representations from SAM [145] and CLIP [119]. Insights & Discussion. These methods demonstrate the limitations of using only 2D or 3D representations and propose strategies for integrating both modalities. Combining multiview images and 3D structures enhances performance and robustness in 3D visual grounding systems.

Year Method Alignment Technique 2023 Chat-3D [150] Multi-modal Transformer 2023 Chat-Scene [151] Multi-modal Transformer

- 2023 3D-LLM [152] Q-Former-liked module

- 2023 GPT4Point [146] Q-Former-liked module

- 2024 LL3DA [37] Q-Former-liked module

- 2023 LEO [147] LLaVA-liked module

- 2024 Scene-LLM [148] LLaVA-liked module

- 2024 LLaVA-3D [28] LLaVA-liked module

- 2025 3D-LLaVA [153] LLaVA-liked module TABLE VII: Comparison in alingment methods.

B. 3D Scene Reasoning and Question Answering (QA)

3D scene reasoning and QA require models capable of processing 3D representations—such as point clouds, meshes, neural radiance fields, or multi-view RGB-D inputs—and generating natural language responses grounded in the spatial and semantic structure of the environment. Current research falls into two paradigms: training-required and training-free. Training-required methods fine-tune MLLMs, typically via Q-Former [37, 146] or projection-layer modules [147, 148]. Training-free methods use frozen MLLMs with progressive prompting [11] and chain-of-thought reasoning [11, 149].

1) Training-required: Training-required studies can be classified into three categories: ① Alignment approach: These methods focus on aligning 3D features with language modalities. ② Training efficiency: Aiming to reduce complexity and improve convergence. ③ 3D Representation: Expanding beyond conventional 3D representations to scene graphs, 3DGS [154, 155], etc.

The next sections elaborate on each category, summarizing current advancements in multimodal spatial reasoning for 3D.

① Recent methods focus on aligning 3D scene features with MLLM feature spaces. Early works [150, 151] use 3D detectors to extract object-level representations, which are aligned with text features using 3D-text paired data, enabling MLLMs to leverage prior knowledge. However, reliance on 3D detectors can be a bottleneck. To address this, inspired by Q-Former [156], recent works [37, 146, 152, 157] integrate similar designs into 3D MLLMs for more complex reasoning. For example, 3UR-LLM [157] uses a 3D compressor to condense 3D features into compact vision tokens and a 3D query fusion mechanism to select high-confidence queries, improving reasoning robustness.

Besides Q-Former, several methods [28, 147, 148, 153] are inspired by LLaVA. These approaches use a projection layer to align the feature space with LLMs, enabling them to process 3D inputs and leverage their spatial reasoning capabilities. For example, Scene-LLM [148] employs a two-stage strategy, training a projection layer with conceptual annotations while keeping the LLM frozen. An overview of these alignment techniques is presented in Table VII.

② Beyond improving alignment quality, recent studies [28, 153, 158, 159] note that aligning 3D features with language is time-consuming. To improve efficiency, 3DMIT [158] removes

Year Method Representation Training

- 2024 3DGraphLLM [29] Scene Graph Full Training

- 2025 SplatTalk [160] 3DGS Fine-tuning 2025 GPT4Scene [161] BEV Zero-shot / Fine-tuning

- TABLE VIII: Comparison of multimodal spatial reasoning methods with diverse 3D representations.

the alignment step by focusing on instruction tuning for spatial understanding. LLaVA-3D [28] retains LLaVA’s 2D multimodal capabilities by constructing 3D patches and using

- 3D-aware positional encoding. Inst3D-LMM [159] introduces multi-task instruction tuning, enabling adaptation to various spatial reasoning tasks without task-specific fine-tuning.

③ Recent works [29, 160, 161] focus on diverse 3D representations, including 3D scene graphs, 3DGS, and BEV. 3DGraphLLM [29] creates a learnable 3D scene graph to enhance spatial reasoning by utilizing richer structural information. SplatTalk [160] integrates language features from RGB images into a unified 3DGS [154] representation, supporting spatial reasoning. GPT4Scene [161] improves reasoning by reconstructing BEV images from 3D scene videos and establishing a consistent mapping between local views and global scene structure. A comparison of these 3D representations is provided in Table VIII.

Insights & Discussion. Efforts to enhance 3D spatial reasoning in MLLMs focus on modality alignment, training efficiency, and exploring alternative 3D representations. However, challenges remain: ① Training 3D-aware models is computationally intensive due to complex data and architectures. ② The lack of large, diverse, and well-annotated 3D datasets limits the effectiveness of supervised training. ③ The absence of transparent reasoning mechanisms hinders interpretability and understanding of model decisions. Addressing these limitations could further advance MLLMs for spatial reasoning.

2) Training-free Methods: Training-free methods [11, 30, 149, 165] leverage the prior knowledge in MLLMs for multimodal spatial reasoning without the need for fine-tuning. These methods explore various prompting strategies to facilitate interpretable spatial reasoning. Some works [11, 149] use MLLMs to extract semantic object attributes and apply the chain-of-thought mechanism, prompting sequential reasoning. SpatialPIN [11] is a modular framework that employs progressive prompting to decompose and reconstruct explicit 3D representations, enhancing spatial reasoning. Agent3DZero [30] introduces a Set-of-Line strategy for selecting and analyzing multiple viewpoints, improving spatial reasoning while reducing memory and computation. LLM-TPC [165] employs a Think-Program-reCtify loop to bridge 3D visual perception and reasoning, improving reliability through iterative self-correction.

Insights & Discussion. These training-free methods utilize MLLMs to summarize and refine spatial information through diverse prompting strategies. Despite their success, they have limitations: ① They depend on the quality of the MLLMs used, and deficiencies in these models may hinder performance on some tasks. ② Some methods involve complex inference steps,

reducing processing speed and making them less suitable for real-time applications.

C. 3D Generation with Spatial Reasoning

3D generation [166, 167] has advanced rapidly, particularly with the integration of LLMs and multimodal reasoning systems. Scene-level and program-level generation demand strong spatial reasoning capabilities. These tasks can be categorized into two aspects: ① 3D Layout Generation: Generating spatially reasonable indoor layouts from natural language or multi-turn dialogues. ② 3D Generation as Program: Treating 3D content generation as a programmatic task, where spatial reasoning is framed as executable program generation.

1) 3D Layout Generation: Given the complexity of 3D scene generation [168–170], researchers often use MLLMs for initial 3D layout generation, followed by scene-level synthesis. Figure 6 presents a qualitative comparison of representative 3D scene generation approaches, showcasing variations in geometric fidelity, texture quality, and semantic consistency across different methods. Approaches can be broadly categorized based on how MLLMs are integrated into the layout pipeline:

### ① Direct Guidance for Scene Synthesis via LLMs:

MLLMs directly generate spatial configurations or layout instructions, translating high-level descriptions into structured commands for scene elements, such as furniture arrangement and room dimensions. However, this direct mapping can lead to implausible configurations, like overlapping objects. Methods like LayoutGPT [5] and HOLODECK [163] address this by incorporating optimization-based solvers or inferring spatial relational constraints.

### ② Indirect Guidance for Scene Synthesis via LLMs:

Indirect guidance uses MLLMs to extract semantic knowledge (e.g., object relationships or contextual constraints) to guide subsequent 3D modeling. For instance, Diorama [138] generates a scene graph defining object relationships, while the MLLM retrieves multimodal 3D shapes. Approaches like LayoutGPT [5] use programmatic reasoning to generate spatial layout specifications, while HOLODECK [163] enhances this with optimization techniques for physical realism. Iterative methods, such as I-Design [164] and Generation Agents [171], introduce multi-agent systems for step-by-step refinement. LLPlace [172] supports real-time interactive layout refinement through a conversational interface, and Chat2Layout [173] combines VQA with visual prompting, enhancing spatial layout reasoning.

Insights & Discussions. The primary approaches either generate positions directly or create intermediate representations like scene graphs. Both paradigms leverage MLLMs for semantically coherent and physically feasible 3D environments. Future advancements in MLLMs could enhance both numerical accuracy and formatting capabilities.

2) 3D Generation as Program: Building on advances in MLLM-based code generation (e.g., Cursor [174] and GitHub Copilot [175]), recent work treats 3D synthesis as procedural program generation, where geometry and layout are specified by code. As shown in Fig. 7, a 3D model can be described by a code snippet, leveraging MLLMs’ structured reasoning and

[Figure 9]

- Fig. 6: Some comparative examples of 3D generations, such as input conditions (e.g., text or image), and outputs from different approaches[138, 162–164], showcasing variations in geometry, texture, and semantic coherence.

[Figure 10]

|import cadquery as cq w0 = cq.Workplane('YZ', origin=(-14, 0, 0))<br><br>r = w0.workplane(offset=17 / 2).moveTo(4,-73.5).box(104, 53, 17)<br><br>.union(w0.sketch().segment((-78, 23), (2,-55)).segment((40,-17))<br><br>.arc((42,-24),(48,-30)).segment((48, 5)).segment((61, 5))<br><br>.segment((78, 22)).segment((-2, 100)).close().assemble()<br><br>.push([(0, 22)]).circle(50, mode='s').finalize().extrude(29))|
|---|

- Fig. 7: A demo for programming 3D object representation. The left is a CAD model, and the right is the corresponding code segment of CAD Query [33].

require deep spatial reasoning, precise geometric control, and integration with downstream tools. While directly generating 3D representations is challenging, using MLLMs for 3D content generation via programming harnesses their full spatial reasoning potential. Programmatic generation is also more controllable, making it better suited for real applications.

V. MULTIMODAL SPATIAL REASONING IN EMBODIED AI

Embodied AI is regarded as a crucial path toward AGI [185]. The rapid progress of MLLMs positions them as promising candidates for the core reasoning module of embodied agents. Many of the core intelligences expected of embodied agents—such as geometric reasoning, navigation, and perspective-taking—fundamentally rely on spatial reasoning capabilities as their foundation [186–188]. As demonstrated in Fig. 8, in this section, we focus on the multimodal spatial reasoning capabilities of MLLM-based embodied agents within the context of current mainstream tasks, including VisionLanguage Action (VLA), Vision-and-Language Navigation (VLN), and other embodied AI tasks.

constraints. Current approaches target three output formats: ① Blender scripts, ② CAD parametric programs, and ③ meshgeneration pipelines.

- ① Blender is the most common software in 3D model-

ing and animation, supporting operations via its API and Python code. The following methods utilize MLLMs’ spatial reasoning for programming outputs. 3D-GPT [32] introduces a training-free framework where an LLM interprets natural language commands and generates Blender scripts to construct 3D scenes, unlocking the potential of MLLMs in spatial programming. SceneCraft [176] proposes a dualloop optimization system: an inner loop refines scenes using MLLM feedback, while the outer loop accumulates spatial knowledge across iterations, enabling self-evolving capabilities. SceneMotifCoder [177] introduces “visual programs” structured code representations extracted from example-based demonstrations.

- ② In addition to Blender, other works extend spatial

reasoning into CAD modeling. CAD-GPT [178] enhances spatial reasoning by integrating spatial tokens and positional embeddings, enabling accurate generation of CAD sequences from images or text. CAD2PROGRAM [179] converts 2D engineering drawings into executable Python scripts using MLLMs. CAD-Recode [33] maps point cloud data into CadQuery scripts via a lightweight encoder and pre-trained MLLM backbone. CAD-LLaMA [180] designs a parametric language to better utilize MLLMs’ spatial knowledge.

- ③ Other work focuses on general mesh generation using

A. Multimodal Spatial Reasoning in VLA Models

VLA models generate executable actions from multimodal inputs—typically visual observations and language instructions—using vision-language foundation models as their backbone. These systems often involve intermediate reasoning steps, either implicit within the architecture or explicit through modular design. Pioneering works such as OpenVLA [189] and π0 [190] adopt an end-to-end paradigm, training VLMs as reactive policies to predict low-level control actions from large-scale demonstrations. Others [45, 191] decompose tasks into natural language sub-tasks executed by reactive controllers or lower-level VLAs, while some frameworks introduce intermediate stages like affordance or goal-state prediction followed by motion planning for action generation.

Regardless of the control representation, spatial reasoning remains central to these systems. Research efforts to improve spatial understanding in VLAs generally follow three directions: ① integrating spatially informative sensor modalities (e.g., depth, point clouds) to enrich spatial context; ② adopting multi-task pre-training or co-training schemes that implicitly encourage spatial reasoning; and ③ incorporating

a programmatic approach. ShapeLib [181] guides LLMs in constructing libraries through a hybrid human-AI workflow.

Insights & Discussions. These works reflect the expanding scope of MLLMs in tackling complex, real-world tasks that

###### Vision-Language-Action (VLA) Vision-and-Language Navigation (VLN) Other Embodied Tasks

Embodied QA

Visual Environment Understanding

Spatially Informative Modalities

Planning and Navigating

[Figure 11]

[Figure 12]

[Figure 13]

| |
|---|

Geometric Reasoning

[Figure 14]

Object Localization

| |
|---|

[Figure 15]

Q: How can I walk to the TV? A *: Walk straight ahead and turn right at the end.

| |
|---|

Q: Can another cup be placed on the decorative shelf? A *: No, it can’t.

Q: Where is the vase? A *: On the table above the radiator.

Embodied Grasping

Human Intention Interpretation

Multi-task Pre- and Co-training

Human: What is beneath the mirror, next to the cot?

[Figure 16]

3D object detection

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

2D pointing

[Figure 22]

Agent: It is a dresser with many drawers.

[Figure 23]

[Figure 24]

Grasp the chocolate to the plate.

Human: Walk to the dresser and face the mirror.

Grasp point and angle prediction

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

Agent: Understood.

[Figure 31]

[Figure 32]

Grasp the apple to the pot.

Explicit Reasoning

Embodied World Model

Path Planning and Navigation

[Figure 33]

[Figure 34]

[Figure 35]

Put the brown mug in the bottom shelf.

[Figure 36]

The brown mug is in front of the shelf and should be placed to the bottom.

Fig. 8: Spatial reasoning in embodied tasks, such as VLA [6, 44, 182], VLN [183, 184] and other embodied tasks [47].

- TABLE IX: Comparison of 3D-enhanced VLA methods.

and 3D point cloud features from a point encoder as inputs to an action expert for prediction. SpatialVLA [193] encodes 3D information into 2D observations using 3D-aware positional encodings derived from monocular depth predictions. BridgeVLA [194] employs dual-phase training: pre-training a VLM for 2D heatmap-based object localization and fine-tuning with multi-view orthographic projections of 3D point clouds to generate action trajectories.

✓indicates the feature is present, and ✗indicates it is absent.

SpatialEncoding

GoalGeneration

3DPerception

PointClouds

DepthMaps

Image source:

- [1] Habitat-Matterport 3D Dataset (HM3D): 1000 Large-scale 3D Environments for Embodied AI
- [2] Bridgedata v2: A dataset for robot learning at scale
- [3] https://tesseractworld.github.io/

- [4] 3D-VLA
- [5] Gemini Robotics
- [6] Chat-VLA2

Method

Training Strategy

3D-VLA [44] ✓ ✓ ✓ ✓ ✗ Diffusion-aligned PointVLA [192] ✓ ✗ ✓ ✗ ✗ Action expert fusion SpatialVLA [193] ✓ ✓ ✗ ✗ ✓ Monocular depth-based encoding BridgeVLA [194] ✓ ✗ ✓ ✗ ✓ Dual-phase (2D pre-train + 3D fine-tune)

Insights & Discussion. These approaches show promise for action prediction with richer spatial perception, but challenges remain. A key limitation is the scarcity of large-scale datasets compared to vision–language corpora [192, 194], motivating synthetic data [44] or imputing missing modalities with pretrained models (e.g., SpatialVLA [193]). Yet such approximations often underperform. Moreover, models trained at scale on 2D vision–language data still lead overall [6, 45, 190], indicating that fully leveraging extra modalities will require targeted pre-training and more data-efficient architectures.

explicit reasoning steps. The following subsections review representative methods in each direction and discuss their respective advantages and limitations.

1) Spatially informative input modalities: Several studies enhance spatial understanding in VLA models by incorporating spatially informative modalities such as depth maps and 3D point clouds, as shown in Table IX. These additional inputs compensate for the limitations of 2D visual data, which often lack the geometric cues needed for reasoning about physical interactions in 3D space. 3D-VLA [44] enhances a language model with 3D perception and goal generation by introducing interaction tokens for objects, locations, scenes, and actions. It aligns the language model with diffusion models that generate goal images, depth maps, and point clouds from instructions. PointVLA [192] combines 2D image features from a VLM

2) Multi-task Pre- and Co-training: Another major approach to enhance spatial understanding in VLA models is to modify the training regime to include auxiliary tasks that implicitly encourage spatial reasoning, such as embodied question answering or 3D bounding box detection, as in Table X. This is typically achieved through pre-training or co-training frameworks that share representations across related spatial tasks. The concept is first explored in RT-2 [195], which jointly trained a VLM on visual question answering and robot

- TABLE X: Comparison of multi-task pre- and co-training strategies for VLA models. ✓indicates the feature is present, and ✗indicates it is absent.

Multi-TaskCo-train

Curriculum/Stage

TrajectoryPred.

EmbodiedQA

3DTasks

Method

RT-2 [195] ✓ ✗ ✗ ✓ ✗ Gemini Robotics [6] ✓ ✓ ✓ ✓ ✓ π0.5 [45] ✓ ✗ ✓ ✓ ✓ ChatVLA [182] ✓ ✗ ✗ ✓ ✓ Magma [196] ✗ ✗ ✗ ✓ ✗

action prediction within a shared token space. Building on this idea, recent large-scale models like GEMINI ROBOTICS [6] and π0.5 [45] employ multi-stage co-training pipelines. GEMINI ROBOTICS [6] adopts a two-stage procedure: the base VLM is first pre-trained on tasks including trajectory prediction, multi-view correspondence, and 3D bounding box detection, yielding the embodied reasoning model Gemini-ER capable of few-shot control through in-context learning. The model is then fine-tuned with an action decoder that outputs low-level control commands for complex manipulation tasks. Similarly, π0.5 [45] pre-trains its VLM backbone on a mixture of tasks such as visual question answering, object localization, sub-task prediction, and discrete action generation. During post-training, an additional action head is introduced for continuous control prediction, followed by fine-tuning for both continuous control and sub-task reasoning.

ChatVLA [182] introduces a two-stage curriculum where the model first learns control from robot data, then training examples from other tasks, such as VQA, are gradually introduces to preserve alignment with pre-trained VLM representations. It also adopts a Mixture of Experts architecture with task-specific heads to avoid task interference. Magma [196] proposes to bridge the gap between vision-language and action data via surrogate tasks that require predicting actionable 2D annotations—Set-of-Mark and Trace-of-Mark. This enables joint training on diverse datasets across digital and physical domains using the same output representation.

Insights & Discussion. Pre- and co- training on spatial reasoning tasks is an effective way to enhance the generalization capabilities of VLA models. However, this approach doesn’t come without its challenges. It requires access to large and diverse datasets, and carefully balancing multiple training objectives. Still, when these challenges are addressed, it remains a core strategy for building capable VLA models.

3) Explicit Reasoning: A third line of research enhances spatial reasoning in VLA models by introducing explicit reasoning steps during action generation. Unlike reactive policies [139, 189, 190] that directly map inputs to actions, these models incorporate structured intermediate representations and multi-step reasoning to interpret spatial relations and plan subtasks before executing actions.

ECoT [197] trains VLA models to generate step-by-step

reasoning chains grounded in the scene and robot state prior to action prediction. These chains include high-level plans, subtasks, object locations, and low-level motions, improving both generalization and interpretability. Chat-VLA2 [46] builds on ChatVLA by adding a reasoning-following module that aligns generated actions with the backbone’s internal reasoning, yielding better performance on multi-step spatial tasks. Chainof-Affordance [198] introduces an affordance-based reasoning process that decomposes tasks into four stages: identifying target objects, selecting grasp points, locating placement regions, and planning trajectories. These affordances, generated at inference time, guide the policy model’s action selection. Similarly, RT-Affordance [199] proposes a hierarchical VLA where action generation is conditioned on affordance plans. An affordance prediction model first generates key poses from images and task descriptions, which then guide a reactive VLA to produce low-level control actions.

Insights & Discussion. Reasoning-augmented models improve robustness, generalization, and interpretability in spatial tasks by explicitly modeling intermediate steps such as object selection, spatial relations, and action planning. This structured reasoning helps policies handle novel objects, scenes, and instructions more effectively than purely reactive baselines. While early methods introduced substantial inference overhead, newer systems mitigate this through selective reasoning and asynchronous pipelines. These trends suggest that the benefits of explicit reasoning can be retained without prohibitive latency, making such models increasingly practical for realworld deployment.

4) Multimodal Spatial Reasoning in Vision Language backbone: Many current VLA models are fine-tuned from VLMs or use them as backbones. These VLAs are claimed to effectively inherit the prior knowledge of these pre-trained models. To quantitatively assess the potential of the upstream VLMs for robotics tasks, we collected open-source VLMs that have been used in VLAs and evaluated them on spatial reasoning benchmarks relevant to embodied scenarios. Specifically, OpenVLA [189] is fine-tuned from Prismatic [200], π0 is finetuned from PaliGemma [201], TraceVLA [140] is fine-tuned from Phi-3-Vision [202], and DexVLA [139] uses Qwen-2VL [203] as its backbone. As for the benchmarks, Embodied Reasoning QA (ERQA) [6] is a benchmark specifically designed for evaluating VLMs in embodied environments. It tests the VLM’s ability to handle embodied tasks. On the other hand, SpatialEval [93] and SPACE [204] are benchmarks that assess the more fundamental and conventional spatial reasoning abilities of VLMs, such as the ability to judge relative spatial positions and distances. Both of these capabilities are crucial for robotics. Therefore, we conducted experiments by testing several VL backbones used in VLA on these benchmarks. As shown in the Tab. XI, it is evident that these backbones exhibit certain spatial reasoning abilities. This is also why these models can achieve strong performance in downstream applications after fine-tuning on robotic datasets.

B. Multimodal Spatial Reasoning in VLN Models

VLN [205] is a cooperative multimodal task where an agent navigates 3D environments by following human instructions

###### Benchmark Prismatic PaliGemma Qwen-2-VL Phi3-Vision

ERQA [6] 32.25 27.25 32.50 34.00 SpatialEval [93] 32.13 29.86 26.80 46.46 SPACE [204] 23.75 17.00 18.75 26.25

- TABLE XI: Embodied-AI-related benchmark results across different VLMs. Note that SpatialEval is tested using VTQA mode(with Vision-Text input).

and communicating in context under ambiguity. It involves four key components: visual perception, language understanding, decision-making, and navigation execution—all requiring strong spatial reasoning. During perception, the agent must localize itself, interpret spatial relationships between objects, and plan an efficient route. Finally, it executes the navigation plan based on these spatial decisions.

1) Visual Environment Understanding and Generalization: For a VLN agent, it is crucial to perceive and interpret its surroundings, anticipate how actions alter the environment, and align perception and decision-making with natural language instructions. This requires understanding spatial arrangements, localizing itself in 3D space, estimating distances between targets and landmarks, retaining spatial information, and tracking environmental changes over time. These abilities collectively depend on strong spatial reasoning, which underpins success in complex vision-and-language navigation tasks.

Existing embodied scene perception methods often rely on 3D or 2.5D data to enhance spatial awareness, as summarized in Tab. XII. To better utilize visual inputs, many approaches explicitly preserve spatial features through multiview perception, depth images, or scene graphs. NaviLLM [207] leverages multiview images to capture all reachable viewpoints from the current position and constructs task-specific schemas for LLMbased action generation. Cai et al. [127] propose SpatialBot, which uses a depth API to query geometric information from the environment and feed it back into the model, strengthening spatial understanding. ConceptGraphs [206] builds an open-vocabulary 3D scene representation by associating 2D foundation model outputs across multiple views.

Beyond visual encoding, another research direction focuses on narrowing the semantic gap between natural language and 3D scene understanding. Spartun3D-LLM [34] integrates a 3D-aware LLM with a situated spatial alignment module to better link 3D visual representations with corresponding textual descriptions. Similarly, Wang et al. [208] introduce a 3D representation model for embodied tasks that predicts novel views and BEV maps at multiple scales, aligning multi-scale feature fields with multi-granularity language representations.

Beyond scene understanding, maintaining environmental memory and tracking temporal changes are equally important. Hong et al. [35] propose GSA-VLN, where agents dynamically update parameters, leverage long-term memory, and adapt to both environments and diverse user instructions. Similarly, Yang et al. [209] present 3D-Mem, a memory architecture that encodes multi-view 3D snapshots to accumulate and retrieve spatial information for long-term perception and reasoning.

Insights & Discussion. Accurate perception, robust spatial

Multi view image Depth image Point clouds Bev feature

Scene Graph Topology Map

[Figure 37]

Efficient 3D Representation

16

#### Multimodal Input Text-3D Scene Memory Alignment

Text

[Figure 38]

[Figure 39]

Point Clouds

Scene Graph

[Figure 40]

[Figure 41]

Multi-view image

[Figure 42]

This is an indoor living room scene. The space includes walls, a ceiling, and a floor, with furniture such as a central table, chairs in front of it, and additional seating on the sides. ……

[Figure 43]

Depth image

BEV Map

Fig. 9: Visual environment understanding in VLN tasks. Current methods take text, point clouds [34], multi-view images [207], RGB-D images [127, 206, 208] as inputs and align them with 3D scene representations, while maintaining structured memories such as scene graphs [206] and BEV maps [208] for effective spatial reasoning.

[Figure 45]

[Figure 46]

[Figure 48]

reasoning, and generalization across diverse visual scenes are fundamental for VLN agents. As shown in Fig. 9, recent work emphasizes structured 3D representations, such as scene graphs, BEV maps, and multiview memory, as effective tools linking perception to reasoning and planning. A key challenge remains the alignment of visual features with linguistic inputs, especially under unfamiliar views or domain shifts.

[Figure 49]

2) Human Intention Interpretation and Instruction Comprehension: VLN agents are required to comprehend natural language instructions provided by humans within specific situational contexts to complete navigation tasks. This involves cor rectly interpreting spatial expressions such as “left,” “up,” and “front,” and developing the ability to reason spatially about object locations, directions, and movements [8]. To facilitate efficient instruction understanding, a common strategy is to incorporate auxiliary modalities into the input. LL3DA [37] encodes 3D point clouds and leverages an attention mechanism to aggregate contextual information from both the scene and human interactions.

In addition, improved VQA paradigms can further enhance an agent’s instruction comprehension. AutoSpatial [36] applies a hierarchical two-round VQA strategy during training, achieving both global and detailed understanding of scenarios, which demonstrates more accurate spatial perception.

Moreover, certain methods, such as affordance prediction, have been introduced to improve the model’s ability to attend to fine-grained visual details under human instructions. Yuan et al. [210] proposed RoboPoint, a vision-language model tailored for predicting spatial affordances from relational language inputs. The model predicts precise action points that comply with spatial and physical constraints, thereby facilitating subsequent action execution.

Insights & Discussion. Recent work highlights the benefits of auxiliary modalities, hierarchical reasoning, and affordance modeling in improving instruction understanding. Multi-round VQA and affordance prediction enhance fine-grained grounding, while attention-based fusion with human interactions supports contextual comprehension. Future advances may rely on tighter integration of spatial perception and language reasoning, along with better generalization to diverse instructions and complex real-world tasks.

Year Method Input Backbone Highlights

- 2024 ConceptGraphs [206] RGB-D image LLaVa Constructs open-vocabulary 3D scene graphs

- 2024 NaviLLM [207] Multi-view RGB image Vicuna-7B-v0 Uses schema-based instruction to adapt LLMs

- 2025 Spartun3D-LLM [34] Point Cloud GPT4o Integrates a 3D-based LLM with a spatial alignment module that links 3D

objects and relations to text, bridging the 3D-text gap

- 2025 g3D-LF [208] RGB-D image Vicuna-7B-v0 Proposes generalizable 3D-language feature fields 2025 SpatialBot [127] RGB-D image QWen1.5-0.5B Introduces depth API to retrieve geometric information

TABLE XII: Comparison of recent multimodal spatial reasoning methods in embodied scene understanding.

TABLE XIII: Comparison of path planning and navigation methods for VLN agents. ✓indicates feature presence.

HierarchicalPlanning

HallucinationMitig.

Mapping/Pre-map

SpatialReasoning

CoTReasoning

DomainAdapt.

Method

NavVLM [38] ✓ ✗ ✗ ✗ ✗ ✗ SpatialCoT [211] ✓ ✓ ✗ ✗ ✗ ✗ NavCoT [39] ✓ ✓ ✓ ✗ ✗ ✗ FlexVLN [212] ✓ ✓ ✓ ✓ ✗ ✗ NavA3 [213] ✓ ✗ ✗ ✗ ✓ ✗ TopV-Nav [214] ✓ ✗ ✗ ✗ ✗ ✓ BrainNav [215] ✓ ✗ ✗ ✗ ✓ ✓

3) Path Planning and Navigation for VLN Agents: VLN agents must combine perception, reasoning, and planning to execute goal-directed navigation from natural-language instructions, as in Table XIII. LLMs often serve as the highlevel planners in these systems. NAVVLM [38] employs a VLM as the cognitive core, interpreting language goals and guiding exploration through semantic understanding of the environment. To enhance spatial reasoning, SPATIALCOT [211] introduces bi-directional spatial coordinate alignment and Chain-of-Thought grounding, improving reasoning accuracy and interpretability.

Addressing domain adaptation, NAVCOT [39] uses parameter-efficient adaptation to enable self-guided navigation, generating coherent reasoning chains aligned with downstream planning. To reduce hallucinated plans, FLEXVLN [212] validates LLM-generated guidance through an auxiliary MLLM, ensuring action feasibility. For longhorizon tasks, NAVA3 [213] adopts a hierarchical framework: a reasoning VLM identifies target regions, and a pointing VLM performs fine-grained localization via spatial affordances.

Mapping-based approaches further improve navigation. TOPV-NAV [214] constructs adaptive top-view maps using visual prompts, providing structured spatial priors for reasoning. BRAINNAV [215] integrates dual maps (coordinate and topological) and dual orientations (relative and absolute), enabling real-time navigation with dynamic scene updates.

Insights & Discussion. Recent methods enhance VLN agents by combining LLM-based planning with spatial grounding, domain adaptation, and hallucination mitigation. Structured spatial priors further support real-time reasoning. Future efforts should unify spatial perception and language reasoning for generalizable, low-supervision navigation.

TABLE XIV: Comparison of representative methods for Embodied Question Answering (EQA). ✓indicates the method supports or explicitly incorporates the feature.

ModularPercept.

3DSceneGraph CoTReasoning

Open-Vocab

RL

Method

Majumdar et al. [40] ✓ ✗ ✗ ✗ ✗ Tan et al. [216] ✗ ✓ ✗ ✗ ✗ Hao et al. [41] ✗ ✗ ✓ ✗ ✗ Zhao et al. [217] ✗ ✗ ✓ ✓ ✓

C. Multimodal Spatial Reasoning in Other Embodied Tasks

1) Embodied Question Answering (EQA): EQA, first proposed by Das et al. [218], has become a central benchmark in embodied AI and robotics. In this task, an agent receives a natural-language question—e.g., “Is there a sofa in the living room?”—and must explore the environment, gather visual evidence, and provide an answer. The challenge lies in grounding language to spatial perception and reasoning. Majumdar et al. [40] developed an open-vocabulary EQA dataset to evaluate foundation models, revealing that current systems struggle with spatial queries requiring object-level and scene-level understanding. To improve spatial reasoning, Tan et al. [216] introduced a 3D scene graph as an external memory, enabling the model to retain and reason over spatial layouts across multiple turns, significantly improving multistep QA efficiency. Hao et al. [41] advanced this direction by integrating Chain-of-Thought (CoT) reasoning within the Embosr framework, allowing structured spatial inference across complex 3D scenarios. Zhao et al. [217] further decoupled perception and reasoning by assigning visual understanding to large-scale VLMs and using a lightweight language model, optimized via reinforcement learning, for reasoning. Incorporating a slow-thinking mechanism enhances depth and reliability in spatial reasoning.

Insights & Discussion. EQA task highlights the intricate interplay between language grounding, visual perception, and spatial reasoning in interactive environments. A key insight from recent advances is that bridging the gap between low-level visual inputs and high-level task understanding requires combining the strong perceptual capabilities of foundation models with explicit reasoning mechanisms, such as scene graphs, neural program synthesis, and chain-of-thought prompting. Future efforts may benefit from further aligning spatial representations with language semantics and enhancing the memory efficiency of agents in multi-turn reasoning settings.

2) Embodied Grasping: Robotic grasping in cluttered environments remains difficult due to occlusions and complex object interactions, demanding fine-grained spatial reasoning. THINKGRASP [42] introduces goal-driven language prompts that help identify and prioritize obstructing objects, enabling grasp planning even for heavily occluded targets. FREEGRASP [43] represents objects as discrete keypoints and overlays visual markers to enhance GPT-4o’s zero-shot spatial reasoning. AFFORDGRASP [219] integrates GPT-4o for incontext affordance reasoning, predicting graspable parts and intended functions, which are grounded using VLPart and Grounded-SAM for part-conditioned optimization. Similarly, UNIDIFFGRASP [220] leverages GPT-4o to infer target semantics and functional parts from user input, combining multistage segmentation and diffusion-based sampling for dual-arm grasp generation in complex scenes.

Insights & Discussion. Cluttered environments, frequent object occlusions, and the need to follow strict temporal and spatial action sequences constitute the primary challenges in embodied grasping tasks. In such settings, spatial reasoning plays a particularly critical role. Using visual observations effectively and appropriately integrating the reasoning capabilities of VLMs are key to addressing these challenges.

3) Embodied World Models: Embodied world models simulate the dynamics of physical environments, supporting policy learning, data-driven simulation, and long-horizon planning. However, models relying solely on 2D pixel observations often fail to capture accurate spatial relationships, leading to incomplete scene representations and weak depth or pose estimation. Structurally consistent scene generation is therefore crucial for effective spatial reasoning and world modeling.

EVA [48] integrates a video generation model with a visual–language model, combining reasoning with high-quality video synthesis. TESSERACT [47] simulates temporal evolution in 3D environments, enabling realistic interactions such as object manipulation and drawer opening while maintaining spatial–temporal consistency across RGB-DN sequences. More recently, 3DFLOWACTION [221] predicts object-level scene flow for manipulation and employs GPT-4o [222] to verify task completion by aligning rendered final states with language descriptions, linking physical dynamics with semantic evaluation.

Insights & Discussion. Embodied world models form the foundation for large-scale simulation data used to train embodied agents. Ensuring geometric and spatial consistency in these generated environments is critical for supporting accurate spatial reasoning and realistic embodied intelligence.

VI. SPATIAL REASONING WITH VIDEO AND AUDIO

- A. Spatial Reasoning with Video

Video inherently captures more information about a scene than static images, leading to significant research into the spatial reasoning capabilities of MLLMs. Extending the reasoning abilities from image-based tasks to video-based understanding opens exciting new possibilities. However, accurately reasoning about spatial properties and establishing correspondences in dynamic, temporal scenes remains a persistent challenge. As

proposed by Spatial-R1 [3], seven critical spatial reasoning tasks are essential in this domain: object relative distance, object size estimation, room size estimation, object relative direction, object appearance order, object absolute distance, and object counting.

We systematically review this emerging area and summarize the key characteristics of the existing methods, as shown in Tab. XV. Recent work has explored specialized architectures and training strategies to enhance spatial reasoning capabilities in MLLMs. A representative example is Spatial-R1 [? ], which proposes fine-tuning vision-language models with reinforcement signals grounded in spatial consistency. This training encourages the model to align outputs with the underlying 3D or 2D geometry implied by the video. SpaceR [3] further refines this approach by injecting positional tokens derived from visual object tracking, enabling improved frame-to-frame localization. Other works introduce complementary strategies. R1-Zero-like training [117] builds on reinforcement objectives to penalize spatial hallucinations and reward temporally stable spatial predictions. ST-Think [109] introduces a dualmodality backbone that processes egocentric video using both motion and layout cues, enabling 4D (space-time) reasoning through transformer modules. Similarly, Video-R1 [16] augments the visual encoder with spatial maps derived from frame-wise geometric analysis, and uses spatial alignment loss to preserve inter-frame consistency. LLaVA-ST [108] and VideoINSTA [50] adopt an orthogonal approach: they focus on instruction tuning with spatial-temporal prompts, encouraging zero-shot understanding of video-level concepts like object permanence and navigational intent. These models rely on vision encoders (typically CLIP variants) that preserve spatial resolution via patch-wise tokenization. In Thinking in Space [8], spatial memory is modeled explicitly through a recurrent memory cache, allowing the LLM to recall visual states at earlier timestamps for long-horizon reasoning. A benchmark-centric perspective is introduced by V-STaR [225], which offers a suite of probing tasks to evaluate spatial reasoning across different axes: motion tracking, occlusion recovery, topological layout understanding, and cross-frame object matching. Coarse Correspondence [223] complements this with a strategy that boosts spatial alignment across frames via coarse-to-fine token matching, improving temporal coherence in reasoning chains. Lastly, Aether [224] proposes geometric-aware world modeling through unified token representations that encode both position and object identity, enabling downstream LLMs to simulate spatial transitions with minimal hallucination.

Insights & Discussion. Recent progress in multimodal spatial reasoning demonstrates the growing capability of MLLMs to handle structured space-time understanding. However, challenges remain: models often lose spatial detail due to token compression and lack mechanisms for robust spatial memory. Solutions such as marker-based overlays (as in MPDrivestyle approaches) and coordinate-augmented prompts (as in LocVLM [126]) provide partial remedies, but fall short in generalizing across diverse video domains. Egocentric video in particular poses unique difficulties for multimodal spatial reasoning: distinguishing between agent motion and object

TABLE XV: Comparison of recent multimodal spatial reasoning methods in video QA.

Year Task Dataset Benchmark Method Spatial Components Code

- Arxiv 2024 MCVQA, OEVQA, etc - - VideoLLaMA2 [49] Convolution Connector link ACL 2024 Long Video-QA - - VideoINSTA [50] Content-based Reasoning link

- Arxiv 2024 ScanQA, OpenEQA - - Coarse Correspondence [223] Lightweight tracking model -

- Arxiv 2024 Video-QA VSI-Bench VSI-Bench[8] - - link

- Arxiv 2025 Video-QA Video-R1 - Video-R1 [16] GRPO link

- Arxiv 2025 Depth Estimation - - AETHER [224] - link

- Arxiv 2025 RSTR - V-STaR [225] - - link Arxiv 2025 Video-QA SpaceR-151k - SpaceR [3] Task-Specific GRPO Training link Arxiv 2025 Video-QA Ego-ST Bench Ego-ST Bench ST-R1 [109] Long-CoT and GRPO link

TABLE XVI: Comparison of recent multimodal spatial reasoning methods in audio.

Year Task Benchmark Method Spatial Components Code NeurIPS 2023 Audio-Visual Sound Localization and Detection STARSS23 [51] - - -

- ICML 2024 Audio-QA SpatialSoundQA [52] BAT Spatial Audio Encoder, Curriculum Learning link

- ICML 2025 Audio-QA AQAPHY ACORN [53] Fundamental Physical Phenomena Arxiv 2025 Audio-Visual-QA SAVVY-Bench SAVVY [54] Spatial Tracks and Global Map Construction link

in Tab. XVI. STARSS23 [51] introduces an audio-visual sound event localization and detection (SELD) task, along with the STARSS23 audio-visual dataset to support spatial reasoning for SELD. SpatialSoundQA [52] is the first largescale benchmark focused on spatial audio question answering (Audio-QA). It includes over 21, 000 simulated binaural audio clips rendered in 3D environments, accompanied by diverse questions involving directionality, distance estimation, and multi-source spatial reasoning. Architecturally, the proposed BAT model combines a spatial audio encoder with a large language model (LLM) and employs curriculum learning to gradually enhance the model’s spatial reasoning capabilities. ACORN [53] also addresses Audio-QA by introducing the AQAPHY benchmark. Technically, it improves an LLM’s spatial reasoning by incorporating fundamental physical phenomena such as the Doppler effect, multipath propagation, and spatial relationships. More recently, SAVVY [54] has emerged as a prominent testbed for spatial reasoning that integrates both audio and visual cues, i.e., audio-visual question answering (Audio-Visual-QA). Specifically, SAVVY presents SAVVYBench, which evaluates 3D spatial reasoning in dynamic scenes with synchronized spatial audio, and proposes to enhance spatial understanding by first extracting spatial tracks and then constructing a global spatial map. These benchmarks collectively advance standardized evaluation for audio spatial reasoning and enable quantitative comparison across MLLMs with varying degrees of spatial awareness. It is worth noting that other Audio-QA and Audio-Visual-QA methods, such as SARI [228], Meerkat [229], and EchoInk-R1 [230], are not discussed here as they do not specifically address spatial reasoning.

Spatial Reasoning with Audio

[Figure 50]

Audio Encoder

Thesoundisfrom…

Audio

[Figure 51]

[Figure 52]

Video (optional)

Visual Encoder LLM

[Figure 53]

How would you describe the location of the sound?

Language Encoder

- Fig. 10: Spatial reasoning from audio & video with MLLMs.

motion requires grounded scene representations and persistent memory. While early efforts such as ST-Think and Thinking in Space offer promising architectures, scalable and generalizable spatial world models remain an open research area.

- B. Spatial Reasoning with Audio

Audio spatial reasoning is the process of interpreting spatial cues from sound, such as direction of arrival, source location, and distance, to infer the physical context of an auditory scene. While human listeners effortlessly localize and segregate sounds using binaural cues, current multimodal large language models (MLLMs) have primarily focused on what is heard (the content) rather than where it is heard from [226]. This lack of spatial awareness limits applications such as audiovisual navigation and egocentric perception, where an AI agent must infer where a sound originates to interact effectively with its environment. To bridge this gap, recent research [51– 54, 226, 227] has begun to explore spatial reasoning capabilities by training large-scale multimodal models that learn from audio-only or audio-visual inputs.

Insights & Discussion. Despite recent progress, significant challenges remain for robust audio spatial reasoning. Current models still struggle to generalize in open-world scenarios with multiple, dynamic sound sources. These limitations are

We systematically review this emerging area and summarize the key characteristics of recently proposed methods, as shown

##### Authors Venue/Date Paper Link Code Input Modality

Feng et al. Arxiv 2025 (Mar) Image-Text Imran Kabir et al. Arxiv 2025 (Mar) Video-Text Peiran Wu et al. Arxiv 2025 (Mar) / Video-Text Ziyue Wang et al. Arxiv 2025 (Mar) Image-Text Jonathan Roberts et al. Arxiv 2025 (Feb) Image-Text Mingjie Xu et al. WACV 2025 Graph-Desc/QA/Conv Hongyu Li et al. Arxiv 2025 (Jan) Video-Text(QA) Yang et al. CVPR 2025 Video-Text(QA) Xingrui Wang et al. CVPR 2025 Image-Text Liao et al. Arxiv 2025 (Apl) Video-Text(QA) Chengzu Li et al. Arxiv 2025 (Jan) / Image-Text Huanqia Cai et al. Arxiv 2025 (Feb) Image-Text Siyu Wang et al. AAAI 2025 CAD-Text Navid Rajabi et al. NIPS 2024 Workshop / Image-Text(QA) Chonghao Sima et al. ECCV 2024 Image/Graph-Text(QA) Ivan Majic et al. GeoAI 2024 Image-Text Li Xuan et al. IOTMMIM 24 / Image-Text Yew Ken Chia et al. ACL 2024 Image-Text Xiao Liu et al. ACL 2022 Image-Text Roshanak Mirzaee et al. NAACL 2021 Text Yu-Chuan Su et al. Arxiv 2021(Apr) Image-Text Letitia Parcalabescu et al. ACL 2022 Image-Text Liu et al. TACL 2023 Image-Text Ramakrishnan et al. Arxiv 2024 (Oct) / Image-Text

TABLE XVII: General MLLM: Benchmarks and Datasets

further compounded by the scarcity of large-scale, highquality spatial audio datasets with precise annotations, which makes it difficult to train models that perform well outside of controlled or simulated environments. To bridge these gaps, promising directions include the development of richer data collection pipelines, such as real-world egocentric recordings or improved simulation techniques that better approximate real acoustic conditions. In parallel, more specialized model architectures are expected to emerge to effectively leverage these spatial cues. By addressing both data and modeling challenges, future systems may achieve human-like “spatial hearing”, reasoning not only about what is heard but also where it occurs within complex, dynamic scenes.

VII. BENCHMARKS

Multimodal spatial reasoning enables AI systems to understand and infer spatial relationships within scenes by integrating information from multiple modalities, such as vision and language. Initially, benchmarks and datasets focused on simple scenes and basic spatial relations. However, as multimodal foundation models evolved, the focus shifted to more complex reasoning and cross-modal inference. Before these models, research was constrained to environments with basic spatial tasks, such as determining relative object positions in visual question answering (VQA). With the rise of powerful pretrained models, new benchmarks were developed to address greater openness, richer complexity, and deeper reasoning capabilities. These efforts span domains like panoramic imagery, video, computer-aided design (CAD), and geographic

information systems (GIS), advancing AI systems in scene understanding. Figure 11 illustrates the development of multimodal spatial reasoning benchmarks. This section provides an overview of the evolution of datasets and benchmarks, highlighting key stages, modality types, and domain coverage, with a focus on those from the foundation model era.

A. Early Multimodal Spatial Reasoning Benchmarks

Before the advent of large-scale multimodal foundation models, early research in spatial reasoning relied heavily on datasets focused on natural images paired with textual descriptions. These datasets aimed to tackle basic spatial reasoning tasks, such as object localization and relationship detection.

A pivotal benchmark in this domain is the Visual Genome dataset [231], which provides annotated images and graphs to depict spatial relationships between objects, facilitating imagetext question answering tasks. Another significant contribution is SpatialSense [232], which contains a wide variety of spatial relationships, promoting tasks that involve misclassificationprone scenarios. Similarly, TVQA+ [233] combines video clips with object detection annotations, requiring models to answer questions that involve both spatial and temporal reasoning. The 2.5VRD dataset [234] focuses on fine-grained visual relationship detection using triplet annotations, capturing spatial relationships between objects.

Additionally, the VALSE benchmark [235], though not solely designed for spatial reasoning, includes rich annotations of spatial relationships and actions, providing an excellent resource for evaluating models’ vision-language grounding capa-

###### Audio-Text

###### Other Modal

Image-Text-Graph

Video-Text

###### Audio-Video

###### 3D-Image-Text

###### Audio-Video-Text

2D-Image-Text

Visual Genome Stanford 2016.02

CAD-GPT SJTU 2024.04

###### TVQA+

VSR Cambridge 2022.05

SpatialSoundQA UTEXAS 2024.02

SpatialSense Princeton 2019.08

SpatialVLM DeepMind 2024.01

2.5VRD Google Research 2021.04

VALSE Uni Heidelberg 2021.12

STARSS23 Sony 2023.06

What’s Up UCLA 2023.10

UNC-Chapel Hill 2019.04

SAT BU 2024.12

SpatialEval MSR 2024.06

ST-Align Meituan 2025.01

VIS-Bench NYU 2024.12

GSR-Bench GMU 2024.06

BLINK UPENN 2024.04

SPACE Apple 2024.10

DriveMLLM WHU 2024.11

CityGPT THU 2024.06

SpatialRGPT UCSD 2024.06

SpaceSGG CityUHK 2024.12

MVoT MSR 2025.01

Spatial457 JHU 2025.02

V-STaR QMUL 2025.03

Ego-ST X-Intellignece 2025.03

ERQA DeepMind 2025.03

SpaceR-151k PKU 2025.04

MM-Escape

Video-R1 CUHK 2025.03

VIS-100k SJTU 2025.04

SAVVY UW 2025.06

AQAPHY NIO 2025.06

OmniSpatial THU 2025.06

THU 2025.03

- Fig. 11: The chronological progression of multimodal spatial reasoning benchmarks. Each colored marker represents a distinct benchmark, with hue variations indicating different modality combinations (e.g., image-text-graph, audio-video). The timeline illustrates the evolution of assessment methodologies and the increasing complexity of spatial reasoning evaluation frameworks.

bilities. Further contributions, such as the VSR dataset [236], define explicit spatial reasoning tasks, while datasets like COCO-Spatial in What’sUp [237] examine the limitations of pre-trained models on spatial reasoning. These early benchmarks, while focused on basic spatial cognition, set the foundation for more advanced tasks, sparking further developments in multimodal spatial reasoning for large-scale models.

- B. Image-Text Spatial Reasoning Benchmarks

With the rise of large-scale multimodal foundation models (MLLMs), spatial reasoning tasks have expanded into various domains. This section discusses the evolution of 2D spatial reasoning benchmarks, categorizing them based on task objectives and methodologies.

1) 2D Spatial Reasoning Tasks: 2D spatial reasoning benchmarks evaluate models’ ability to reason about spatial relationships in two-dimensional settings, focusing on tasks like navigation, object localization, and layout generation. A key trend is the integration of multimodal data, combining visual and textual information for enhanced reasoning. For example, DriveMLLM [238] annotates spatial relationships in driving scenarios using question-answer pairs, assessing navigation understanding. SpatialEval [239] provides synthetic images with spatial tasks, such as Spatial-Map and MazeNav, testing relative object positioning in controlled settings. The SPACE benchmark [204] offers both large-scale and small-scale tasks, from layout understanding to viewpoint transformations, evaluating models’ ability to handle diverse spatial challenges.

- 2) Hybrid Approaches and Abstract Representations: Some

datasets explore abstract representations. VSR [236] provides annotations for spatial positional relationships, testing complex spatial reasoning. Datasets like COCO-Spatial [237] introduce spatial tasks that involve context, navigation, and dynamic reasoning. Other benchmarks, such as OmniSpatial [96] and GSR-Bench [240], enhance real-world relevance, offering comprehensive evaluations in areas like autonomous driving and robotics. OmniSpatial tests tasks like dynamic reasoning, traffic analysis, and geometric decomposition, reflecting realworld spatial complexities.

- 3) Insights & Discussion: 2D spatial reasoning datasets

have evolved from simple image-text pairs to multi-task frameworks evaluating diverse reasoning abilities. Recent datasets emphasize multimodal data, combining visual and textual information for complex reasoning. Although synthetic data accelerates benchmarking, it faces challenges in generalization and real-world applicability. Future benchmarks should integrate dynamic real-world data and hybrid datasets combining synthetic and real data to better cover edge cases and enhance evaluation. These advancements will enable more capable models for autonomous navigation, robotics, and other complex applications.

- 4) 3D Spatial Reasoning Benchmarks: The development

of 3D spatial reasoning datasets has significantly advanced in recent years. Boyuan Chen et al.introduced the first 3D spatial reasoning dataset [15], incorporating depth-aware reasoning into multimodal systems. To evaluate MLLMs on dynamic spatial reasoning tasks, Arijit Ray et al.proposed the SAT dataset [110], which includes simulated 3D scenes for training

and real-world environments for testing. This dataset, using 3D scene simulations, improves model performance on dynamic spatial reasoning tasks through real-world evaluation.

To further address gaps in 3D spatial reasoning, AnChieh Cheng et al.introduced SpatialRGPT-Bench [18], which generates 3D reasoning tasks grounded in 2D scenes. Their pipeline combines instance segmentation and depth estimation to construct tasks such as object size, height, and relative distance estimation using only 2D inputs. Additionally, Xingrui Wang et al.developed the Spatial457 dataset [241] for 6D spatial reasoning, covering 3D localization, orientations, and multi-object relationships, further assessing the performance of MLLMs on these complex tasks.

Insights & Discussion. The introduction of 3D spatial reasoning benchmarks has brought significant advances, especially in data generation. Synthesis-driven annotation methods and automated 2D-to-3D conversion pipelines have alleviated annotation challenges. As tasks evolve, they have shifted from basic orientation and static perception to dynamic scene understanding and multi-perspective reasoning, increasing cognitive complexity. Furthermore, evaluation frameworks have transitioned from simulation-based training to real-world scenario validation, establishing closed-loop paradigms for performance assessment. Despite these advances, challenges remain, particularly in cross-modal alignment and adapting to dynamic scenes, highlighting the need for continued research in these areas.

- C. Video-Text Spatial Reasoning Benchmarks

Recent advancements in video-text spatial reasoning have led to the development of diverse benchmarks aimed at systematically evaluating spatial understanding capabilities. These benchmarks have evolved from fundamental perceptual tasks to more complex spatiotemporal tasks. Current benchmarks increasingly emphasize the integration of temporal and spatial cues, leveraging both synthetic and annotated data to support model training and evaluation. The following sections provide a detailed overview of these benchmarks, categorized by task type and complexity, highlighting their contributions in vediotext spatial reasoning.

1) Fundamental Spatial Perception Tasks: Benchmarks in this category evaluate core spatial perception skills such as object counting, relative direction, and distance estimation. VIS-100K [117] introduces 100,000 video–question–answer pairs spanning six spatial reasoning tasks—object count, relative/absolute distance, relative direction, object size, and room size. Fine-tuning MLLMs on this dataset demonstrates that the GRPO reinforcement algorithm effectively enhances spatial reasoning performance. VIS-BENCH [8] further examines how MLLMs memorize and reason about spatial layouts. Built from 288 annotated indoor videos, it includes 5,000 QA pairs covering eight tasks such as distance, direction, path planning, and order of appearance, offering a detailed analysis of spatial understanding. SPACER-151K [3] expands this scope with 151K samples, including 91K spatial QA pairs and 60K general video understanding examples. Each task incorporates precise spatial metadata (e.g., bounding boxes, temporal indices) and 10×10 grid maps encoding object distributions.

TABLE XVIII: Performance comparison on Video–Text Spatial Reasoning Benchmarks (reported pairs only). Metrics follow the originals: SPATIALRGPT-BENCH—Success Rate; BLINK—Accuracy (spatial subset); SPATIALEVAL—Accuracy (0–1); DRIVEMLLM—Zero-shot Score; SAT—Accuracy on Real/Synthetic.

Benchmark Metric Model Value SPATIALRGPT-BENCH [18]

Success Rate LLaVA-v1.6-34B [242] 43.98 Success Rate GPT-4V [243] 58.14

###### BLINK (spatial) [244]

Acc. LLaVA-v1.6-34B [242] 76.22 Acc. InstructBLIP-Vicuna-7B [245] 55.24 Acc. InstructBLIP-Vicuna-13B [245] 64.34 Acc. Gemini-Pro [246] 67.13 Acc. GPT-4V [243] 72.03 Acc. GPT-4o [243] 76.92

SPATIALEVAL [239] Acc. (0–1) LLaVA-v1.6-Mistral-7B [242] 0.33 Acc. (0–1) LLaVA-v1.6-Vicuna-7B [242] 0.24 Acc. (0–1) LLaVA-v1.6-Vicuna-13B [242] 0.38 Acc. (0–1) LLaVA-v1.6-34B [242] 0.42 Acc. (0–1) InstructBLIP-Vicuna-7B [245] 0.24 Acc. (0–1) InstructBLIP-Vicuna-13B [245] 0.27 Acc. (0–1) Gemini-Pro [246] 0.687 Acc. (0–1) GPT-4V [243] 0.924

###### DRIVEMLLM [238]

Score (ZS) LLaVA-v1.6-Mistral-7B [242] 38.20 Score (ZS) LLaVA-v1.6-Vicuna-7B [242] 38.20 Score (ZS) LLaVA-v1.6-Vicuna-13B [242] 38.20 Score (ZS) LLaVA-ov-7B [247] 22.29 Score (ZS) LLaVA-ov-72B [247] 21.10 Score (ZS) Qwen2-VL-7B [248] 21.17 Score (ZS) Qwen2-VL-72B [248] 20.11 Score (ZS) Qwen-VL [249] 36.50 Score (ZS) mPLUG-Owl2 [250] 33.90 Score (ZS) InstructBLIP-Vicuna-7B [245] 42.80 Score (ZS) InstructBLIP-Vicuna-13B [245] 42.80 Score (ZS) Gemini-1.5-flash [251] 54.03 Score (ZS) Gemini-Pro [246] 40.10 Score (ZS) GPT-4V [243] 51.70 Score (ZS) GPT-4o [243] 25.63

###### SAT [110]

Acc. (Real) Gemini-1.5-flash [251] 57.60 Acc. (Synthetic) Gemini-1.5-flash [251] 50.00 Acc. (Real) Gemini-1.5-Pro [251] 64.80 Acc. (Synthetic) Gemini-1.5-Pro [251] 49.90 Acc. (Real) GPT-4V [243] 50.70 Acc. (Synthetic) GPT-4V [243] 44.80 Acc. (Real) GPT-4o [243] 57.50 Acc. (Synthetic) GPT-4o [243] 49.40

Rigorous quality control ensures balanced, unambiguous data, establishing a new large-scale benchmark for spatial reasoning in multimodal systems.

2) Advanced Spatiotemporal Reasoning Tasks: These benchmarks extend spatial reasoning to dynamic tasks such as path planning and cross-modal coordination, emphasizing temporal consistency and causal reasoning. ST-ALIGN [108] establishes a unified framework for fine-grained spatiotemporal reasoning with three tasks: Spatial-Temporal Video Grounding (STVG), Event Localization and Captioning (ELC), and Spatial Video Grounding (SVG). It jointly evaluates spatial and temporal localization, advancing beyond datasets focused on isolated spatial or temporal cues. EGO-ST [109] addresses the overlooked role of temporal dynamics by introducing reverse egocentric reasoning. Comprising over 5,000 QA pairs

across four tasks—route description, directional change, landmark transition, and action shift—it systematically evaluates how MLLMs integrate dynamic spatial cues and temporal order. V-STAR [225] targets the gap between object-centric and temporal reasoning. Its core Reverse Spatio-Temporal Reasoning (RSTR) task links “What→When→Where” and “What→Where→When” chains to assess logical consistency, using the Logarithmic Geometric Mean (LGM) metric to jointly measure accuracy, temporal IoU, and spatial IoU. It establishes the first standardized benchmark for comprehensive spatiotemporal reasoning in Video-LLMs. Overall, these datasets advance spatial intelligence evaluation from static spatial perception to dynamic, temporally grounded reasoning—crucial for realistic embodied and video understanding.

3) Mixed-Task Benchmarking: This class of evaluation benchmarks incorporates diverse data sources and tasks of varying difficulty levels to provide a comprehensive assessment of model capabilities. Due to the scarcity of highquality video reasoning data, current MLLMs exhibit limited spatial reasoning capabilities in video contexts. To address this issue, Feng et al.introduced two datasets: Video-R1-COT-165k and Video-R1-260k[16]. The former contains CoT annotated samples generated from both image and video inputs, serving as a cold-start dataset for supervised fine-tuning. The latter is designed for reinforcement learning training, comprising a mix of image and video data to enable models to acquire general reasoning skills from static images and transfer them to dynamic video contexts through a hybrid training strategy. Although only about 8% of the samples in these datasets involve explicit spatial reasoning tasks, the inclusion of complete CoT annotations offers valuable resources for advancing research on spatial reasoning in video-based settings.

Insights & Discussion. Current visual-spatial reasoning benchmarks are advancing from static attribute recognition toward dynamic spatiotemporal coupling, demanding progressively higher spatial cognitive capabilities from models; however, they remain constrained by limitations including prohibitive annotation costs restricting dataset scalability, inconsistent quality in semi-automated multimodal LLM-generated annotations, and overly homogeneous templated data that inadequately fosters profound spatial cognition—necessitating a paradigm shift from isolated data curation to synergistic algorithm-data co-design, from single-modality datasets to multi-source hybrid data frameworks, and from superficial pattern matching to causal inference incorporating physical constraints like gravitational collision dynamics.

- D. Other Modal Benchmarks

Additional multimodal benchmarks extend spatial reasoning beyond vision–language inputs. For audio–visual spatial reasoning, related datasets and evaluation protocols are detailed in Section VI-B. CITYINSTRUCTION and CITYEVAL, released with CITYGPT [107], evaluate spatial reasoning, navigation, and path generation in realistic urban scenes. CAD-GPT [178] introduces a dataset pairing natural language descriptions and single-view images with CAD modeling sequences, enabling multimodal 3D model synthesis and benchmarking.

SGG [252] provides structured scene graphs and 3D point clouds fused with LLM-generated question–answer dialogues, supporting open-vocabulary spatial reasoning across complex visual layouts. Finally, MM-ESCAPE [253] simulates an interactive escape-room environment where models must perform sequential spatial reasoning and actions to exit, offering a novel framework for evaluating goal-driven reasoning in dynamic scenes.

Insights & Discussion. Contemporary multimodal spatial reasoning datasets exhibit a tripartite evolution—progressing from scene-driven construction to task sophistication and evaluation closure—where real-world task demands grow increasingly complex, modalities diversify, and spatial reasoning advances beyond basic directional perception toward causal spatial inference chains. Nevertheless, persistent gaps remain in establishing a unified framework that ensures physical plausibility, enables action verifiability, and maintains costeffective data curation, indicating considerable scope for advancement in multimodal spatial reasoning data infrastructure.

VIII. CHALLENGES AND FUTURE DIRECTIONS

Multimodal Spatial Reasoning in Egocentric Vision. While existing research on spatial reasoning in MLLMs primarily focuses on third-person perspectives, there is a growing need to explore egocentric vision, where spatial reasoning must occur from the agent’s first-person viewpoint [254–256]. This shift introduces unique challenges, such as the agent’s movement, limited field of view, and the temporally evolving nature of the environment. In egocentric vision, spatial reasoning must account for dynamic changes in both the agent’s position and the environment. Future research should focus on developing MLLMs capable of understanding object relationships from shifting viewpoints, inferring navigation intent, and reasoning about interaction affordances. A promising direction lies in creating models that can more effectively simulate and understand embodied behaviors, leading to more grounded, realworld intelligence.

Multimodal Spatial Reasoning in 3D Vision. Despite progress, current 3D MLLMs face challenges in scalability and interpretability due to the inherent complexity of 3D data. Additionally, the scarcity of large-scale annotated 3D datasets constrains the development of robust models. To address these challenges, future research should focus on the development of unified and efficient 3D representations that are both interpretable and scalable. Furthermore, training strategies that do not rely on large-scale annotated datasets, such as leveraging synthetic data, could offer valuable insights. By exploring the integration of symbolic reasoning into the 3D domain, researchers can ensure better handling of spatial relationships and improve model performance across unseen environments. A key goal should be creating frameworks that combine efficient 3D learning with strong temporal and causal reasoning capabilities to model dynamic spatial environments. Multimodal Spatial Reasoning in Embodied AI. Current methods for spatial reasoning in embodied AI often struggle to generalize to novel environments and are prone to spurious or hallucinated spatial inferences. Explicit reasoning modules,

while improving inter-pretability, tend to increase inference overhead and still fall short in maintaining long-term spatial consistency. To advance this field, future research must focus on closer integration between perception and reasoning, ensuring that spatial models maintain both geometric fidelity and temporal consistency. Additionally, creating world models that combine sensory inputs (e.g., visual, auditory, tactile) with structured scene representations could allow for more robust spatial reasoning in dynamic environments. Scalable training strategies that incorporate symbolic and structured reasoning, along with the ability to perform causal inference over time, will be crucial in achieving long-term success in this area.

Multimodal Spatial Reasoning with Novel Sensors. Emerging sensor technologies such as omnidirectional cameras [69, 70, 257], event cameras, LiDAR, thermal, and radar sensors offer complementary spatial information under challenging conditions like adverse lighting, weather, and high-speed motion [258]. However, these sensors introduce new challenges, including equirectangular distortions, orientation ambiguities, sparse and asynchronous data, and noise in radar and thermal signals. MLLMs, which are typically optimized for perspective RGB images, must evolve to effectively integrate and process these diverse modalities. Future research should focus on developing methods for fusing these heterogeneous sensor data into a unified spatial representation [259], improving both the accuracy and robustness of spatial reasoning. By incorporating causal and temporal reasoning capabilities into sensor fusion, models can better handle dynamic environments and make more informed, context-aware decisions [260]. Moreover, training strategies that leverage both synthetic and real-world sensor data could enhance model generalization across different sensor modalities.

Multimodal Spatial Reasoning Benchmarks. Existing benchmarks are limited in their scope, often suffering from issues such as orientation under-specification, narrow modality coverage, and restricted interaction. To address these limitations, future work should focus on developing more comprehensive benchmarks that span a wider range of modalities and interaction settings. This includes constructing benchmarks that synchronize vision, depth, point clouds, panoramic views, spatial audio, inertial signals, and topological maps, all within a unified coordinate frame with explicit orientation and reference-frame labels. Future benchmarks should also focus on evaluating MLLMs’ ability to perform tasks such as reference, navigation, inspection, and question answering in diverse environments. The development of interpretable evaluation frameworks that can assess both reasoning quality and spatial accuracy, while providing clear guidance for model improvement, will be essential. Additionally, incorporating symbolic reasoning into these benchmarks could allow for the assessment of structured spatial knowledge and enable better handling of complex real-world tasks.

IX. CONCLUSION

Large multimodal reasoning models have gradually emerged as a promising and critical solution toward achieving spatial reasoning capabilities. In this paper, we focus on the

intersection of spatial reasoning and MLLMs. Firstly, based on general spatial reasoning tasks, we systematically review and analyze the existing research from four perspectives: testtime scaling, post-training, model design, and explainability. We then extend the discussion to 3D vision tasks, including 3D visual grounding, 3D scene reasoning and question answering, and 3D generation. Beyond these fundamental tasks, we further explore spatial reasoning in embodied AI, providing reviews and discussions on vision-language navigation, embodied question answering, and related areas. Moreover, spatial reasoning tasks involving emerging modalities such as video and audio are also summarized, which are challenging but crucial to building a comprehensive human-like spatial reasoning system. In addition to methodological aspects, we provide a comprehensive overview of datasets and benchmarks for multimodal spatial reasoning, which constitute the indispensable support for advancements in this field. Through this systematic survey, we aim to establish a solid knowledge foundation and offer new insights to this field-paving the way toward intelligent and reliable multimodal spatial reasoning systems in the era of large models.

REFERENCES

- [1] A. Su, H. Wang, W. Ren, F. Lin, and W. Chen, “Pixel reasoner: Incentivizing pixel-space reasoning with curiosity-driven reinforcement learning,” arXiv preprint arXiv:2505.15966, 2025.
- [2] D. Liu, C. Wang, P. Gao, R. Zhang, X. Ma, Y. Meng, and Z. Wang, “3daxisprompt: Promoting the 3d grounding and reasoning in gpt-4o,” Neurocomputing, vol. 637, p. 130072, 2025.
- [3] K. Ouyang, Y. Liu, H. Wu, Y. Liu, H. Zhou, J. Zhou, F. Meng, and X. Sun, “Spacer: Reinforcing mllms in video spatial reasoning,” arXiv preprint arXiv:2504.01805, 2025.
- [4] Y. Du, T. Fu, Z. Chen, B. Li, S. Su, Z. Zhao, and C. Wang, “Vl-nav: Real-time vision-language navigation with spatial reasoning,” arXiv preprint arXiv:2502.00931, 2025.
- [5] W. Feng, W. Zhu, T.-j. Fu, V. Jampani, A. Akula, X. He, S. Basu, X. E. Wang, and W. Y. Wang, “Layoutgpt: Compositional visual planning and generation with large language models,” Advances in Neural Information Processing Systems, vol. 36, pp. 18225–18250, 2023.
- [6] G. R. Team, S. Abeyruwan, J. Ainslie, J.-B. Alayrac, M. G. Arenas, T. Armstrong, A. Balakrishna, R. Baruch, M. Bauza, M. Blokzijl et al., “Gemini robotics: Bringing ai into the physical world,” arXiv preprint arXiv:2503.20020, 2025.
- [7] F. Shiri, X.-Y. Guo, M. Far, X. Yu, R. Haf, and Y.-F. Li, “An empirical analysis on spatial reasoning capabilities of large multimodal models,” in Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, 2024, pp. 21440–21455.
- [8] J. Yang, S. Yang, A. W. Gupta, R. Han, L. Fei-Fei, and S. Xie, “Thinking in space: How multimodal large

- language models see, remember, and recall spaces,” arXiv preprint arXiv:2412.14171, 2024.
- [9] W. Wu et al., “Mind’s eye of llms: Visualizationof-thought elicits spatial reasoning in large language models,” in Advances in Neural Information Processing Systems (NeurIPS), 2024.
- [10] H. Wu, X. Huang, Y. Chen, Y. Zhang, Y. Wang, and W. Xie, “Spatialscore: Towards unified evaluation for multimodal spatial understanding,” arXiv preprint arXiv:2505.17012, 2025.
- [11] C. Ma, K. Lu, T.-Y. Cheng, N. Trigoni, and A. Markham, “Spatialpin: Enhancing spatial reasoning capabilities of vision-language models through prompting and interacting 3d priors,” arXiv preprint arXiv:2403.13438, 2024.
- [12] Y. Wang, T. Zhou, Z. Peng, X. Li, Y. Chen, and X. Chen, “Visuothink: Empowering lvlm reasoning with multimodal tree search,” arXiv preprint arXiv:2504.09130, 2025.
- [13] I. Kabir, M. A. Reza, and S. Billah, “Logic-rag: Augmenting large multimodal models with visualspatial knowledge for road scene understanding,” arXiv preprint arXiv:2503.12663, 2025.
- [14] X. Xu et al., “Multi-spatialmllm: Multi-frame spatial understanding with multi-modal large language models,” arXiv preprint arXiv:2505.17015, 2025.
- [15] B. Chen, Z. Xu, S. Kirmani, B. Ichter, D. Sadigh, L. Guibas, and F. Xia, “Spatialvlm: Endowing visionlanguage models with spatial reasoning capabilities,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 14455– 14465.
- [16] K. Feng, K. Gong, B. Li, Z. Guo, Y. Wang, T. Peng, B. Wang, and X. Yue, “Video-r1: Reinforcing video reasoning in mllms,” arXiv preprint arXiv:2503.21776, 2025.
- [17] D. Wu et al., “Spatial-mllm: Boosting mllm capabilities in visual-based spatial intelligence,” arXiv preprint arXiv:2505.23747, 2025.
- [18] A.-C. Cheng, H. Yin, Y. Fu, Q. Guo, R. Yang, J. Kautz, X. Wang, and S. Liu, “Spatialrgpt: Grounded spatial reasoning in vision language models,” arXiv preprint arXiv:2406.01584, 2024.
- [19] P. He, Z. Zhang, Y. Zhang, X. Zhao, and S. Peng, “Spatial-ormllm: Improve spatial relation understanding in the operating room with multimodal large language model,” arXiv preprint arXiv:2508.08199, 2025.
- [20] Y. Qi et al., “Beyond semantics: Rediscovering spatial awareness in vision-language models,” arXiv preprint arXiv:2503.17349, 2025.
- [21] S. Chen et al., “Why is spatial reasoning hard for vlms? an attention mechanism perspective on focus areas,” arXiv preprint arXiv:2503.01773, 2025.
- [22] C. Wen, D. Jayaraman, and Y. Gao, “Can transformers capture spatial relations between objects?” in Proceedings of the International Conference on Learning Representations (ICLR 2024), 2024.
- [23] J. Yang, X. Chen, S. Qian, N. Madaan, M. Iyengar, D. F.

- Fouhey, and J. Chai, “Llm-grounder: Open-vocabulary 3d visual grounding with large language model as an agent,” in 2024 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2024, pp. 7694–7701.
- [24] Y. Chen, S. Yang, H. Huang, T. Wang, R. Xu, R. Lyu, D. Lin, and J. Pang, “Grounded 3d-llm with referent tokens,” arXiv preprint arXiv:2405.10370, 2024.
- [25] R. Xu, Z. Huang, T. Wang, Y. Chen, J. Pang, and D. Lin, “Vlm-grounder: A vlm agent for zero-shot 3d visual grounding,” arXiv preprint arXiv:2410.13860, 2024.
- [26] R. Li, S. Li, L. Kong, X. Yang, and J. Liang, “Seeground: See and ground for zero-shot open-vocabulary 3d visual grounding,” arXiv preprint arXiv:2412.04383,

- 2024.

[27] Z. Liu, Y. Wang, S. Zheng, T. Pan, L. Liang, Y. Fu, and X. Xue, “Reasongrounder: Lvlm-guided hierarchical feature splatting for open-vocabulary 3d visual grounding and reasoning,” arXiv preprint arXiv:2503.23297,

- 2025.

- [28] C. Zhu, T. Wang, W. Zhang, J. Pang, and X. Liu, “Llava3d: A simple yet effective pathway to empowering lmms with 3d-awareness,” arXiv preprint arXiv:2409.18125, 2024.
- [29] T. Zemskova and D. Yudin, “3dgraphllm: Combining semantic graphs and large language models for 3d scene understanding,” arXiv preprint arXiv:2412.18450, 2024.
- [30] S. Zhang, D. Huang, J. Deng, S. Tang, W. Ouyang, T. He, and Y. Zhang, “Agent3d-zero: An agent for zero-shot 3d understanding,” in European Conference on Computer Vision. Springer, 2024, pp. 186–202.
- [31] J. Zhou, X. Li, L. Qi, and M.-H. Yang, “Layout-your3d: Controllable and precise 3d generation with 2d blueprint,” arXiv preprint arXiv:2410.15391, 2024.
- [32] C. Sun, J. Han, W. Deng, X. Wang, Z. Qin, and S. Gould, “3d-gpt: Procedural 3d modeling with large language models,” arXiv preprint arXiv:2310.12945, 2023.
- [33] D. Rukhovich, E. Dupont, D. Mallis, K. Cherenkova, A. Kacem, and D. Aouada, “Cad-recode: Reverse engineering cad code from point clouds,” arXiv preprint arXiv:2412.14042, 2024.
- [34] Y. Zhang, Z. Xu, Y. Shen, P. Kordjamshidi, and L. Huang, “Spartun3d: Situated spatial understanding of 3d world in large language models,” arXiv preprint arXiv:2410.03878, 2024.
- [35] H. Hong, Y. Qiao, S. Wang, J. Liu, and Q. Wu, “General scene adaptation for vision-and-language navigation,” arXiv preprint arXiv:2501.17403, 2025.
- [36] Y. Kong, D. Song, J. Liang, D. Manocha, Z. Yao, and X. Xiao, “Autospatial: Visual-language reasoning for social robot navigation through efficient spatial reasoning learning,” arXiv preprint arXiv:2503.07557, 2025.
- [37] S. Chen, X. Chen, C. Zhang, M. Li, G. Yu, H. Fei, H. Zhu, J. Fan, and T. Chen, “Ll3da: Visual interactive instruction tuning for omni-3d understanding reasoning and planning,” in Proceedings of the IEEE/CVF Con-

- ference on Computer Vision and Pattern Recognition, 2024, pp. 26428–26438.
- [38] Z. Yin, C. Cheng et al., “Navigation with vlm framework: Go to any language,” arXiv preprint arXiv:2410.02787, 2024.
- [39] B. Lin, Y. Nie, Z. Wei, J. Chen, S. Ma, J. Han, H. Xu, X. Chang, and X. Liang, “Navcot: Boosting llm-based vision-and-language navigation via learning disentangled reasoning,” IEEE Transactions on Pattern Analysis and Machine Intelligence, 2025.
- [40] A. Majumdar, A. Ajay, X. Zhang, P. Putta, S. Yenamandra, M. Henaff, S. Silwal, P. Mcvay, O. Maksymets, S. Arnaud et al., “Openeqa: Embodied question answering in the era of foundation models,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2024, pp. 16488–16498.
- [41] Y. Hao, F. Yang, N. Fang, and Y.-S. Liu, “Embosr: Embodied spatial reasoning for enhanced situated question answering in 3d scenes,” in 2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2024, pp. 9811–9816.
- [42] Y. Qian, X. Zhu, O. Biza, S. Jiang, L. Zhao, H. Huang, Y. Qi, and R. Platt, “Thinkgrasp: A vision-language system for strategic part grasping in clutter,” arXiv preprint arXiv:2407.11298, 2024.
- [43] R. Jiao, A. Fasoli, F. Giuliari, M. Bortolon, S. Povoli, G. Mei, Y. Wang, and F. Poiesi, “Free-form languagebased robotic reasoning and grasping,” arXiv preprint arXiv:2503.13082, 2025.
- [44] H. Zhen, X. Qiu, P. Chen, J. Yang, X. Yan, Y. Du, Y. Hong, and C. Gan, “3d-vla: A 3d vision-languageaction generative world model,” in International Conference on Machine Learning. PMLR, 2024, pp. 61229–61245.
- [45] P. Intelligence, K. Black, N. Brown, J. Darpinian, K. Dhabalia, D. Driess, A. Esmail, M. Equi, C. Finn, N. Fusai et al., “\pi 0.5: a vision-language-action model with open-world generalization,” arXiv preprint

- arXiv:2504.16054, 2025.

[46] Z. Zhou, Y. Zhu, J. Wen, C. Shen, and Y. Xu, “Visionlanguage-action model with open-world embodied reasoning from pretrained knowledge,” arXiv preprint

- arXiv:2505.21906, 2025.

- [47] H. Zhen, Q. Sun, H. Zhang, J. Li, S. Zhou, Y. Du, and C. Gan, “Tesseract: learning 4d embodied world models,” arXiv preprint arXiv:2504.20995, 2025.
- [48] X. Chi, H. Zhang, C.-K. Fan, X. Qi, R. Zhang, A. Chen, C.-m. Chan, W. Xue, W. Luo, S. Zhang et al., “Eva: An embodied world model for future video anticipation,” arXiv preprint arXiv:2410.15461, 2024.
- [49] Z. Cheng, S. Leng, H. Zhang, Y. Xin, X. Li, G. Chen, Y. Zhu, W. Zhang, Z. Luo, D. Zhao, and L. Bing, “Videollama 2: Advancing spatial-temporal modeling and audio understanding in video-llms,” arXiv preprint arXiv:2406.07476, 2024.
- [50] R. Liao, M. Erler, H. Wang, G. Zhai, G. Zhang, Y. Ma, and V. Tresp, “Videoinsta: Zero-shot long video understanding via informative spatial-temporal reasoning

- with llms,” in Findings of the Association for Computational Linguistics: EMNLP 2024, 2024, pp. 6577–6602.
- [51] K. Shimada, A. Politis, P. Sudarsanam, D. A. Krause, K. Uchida, S. Adavanne, A. Hakala, Y. Koyama, N. Takahashi, S. Takahashi et al., “Starss23: An audiovisual dataset of spatial recordings of real scenes with spatiotemporal annotations of sound events,” Advances in neural information processing systems, vol. 36, pp. 72931–72957, 2023.
- [52] Z. Zheng, P. Peng, Z. Ma, X. Chen, E. Choi, and D. Harwath, “Bat: Learning to reason about spatial sounds with large language models,” arXiv preprint arXiv:2402.01591, 2024.
- [53] W. Wang, A. Nie, W. Zhou, Y. Kai, and C. Hu, “Teaching physical awareness to llms through sounds,” arXiv preprint arXiv:2506.08524, 2025.
- [54] M. Chen, Z. Cui, X. Liu, J. Xiang, C. Zheng, J. Li, and E. Shlizerman, “Savvy: Spatial awareness via audiovisual llms through seeing and hearing,” arXiv preprint arXiv:2506.05414, 2025.
- [55] J. Bai, S. Bai, Y. Chu, Z. Cui, K. Dang, X. Deng, Y. Fan, W. Ge, Y. Han, F. Huang et al., “Qwen technical report,” arXiv preprint arXiv:2309.16609, 2023.
- [56] J. Zha, Y. Fan, X. Yang, C. Gao, and X. Chen, “How to enable llm with 3d capacity? a survey of spatial reasoning in llm,” arXiv preprint arXiv:2504.05786, 2025.
- [57] Q. Ma, R. Yang, B. Ren, N. Sebe, E. Konukoglu, L. Van Gool, and D. P. Paudel, “Cityloc: 6dof pose distributional localization for text descriptions in large-scale scenes with gaussian representation,” arXiv preprint arXiv:2501.08982, 2025.
- [58] X. Zheng, C. Liao, Y. Fu, K. Lei, Y. Lyu, L. Jiang, B. Ren, J. Chen, J. Wang, C. Li et al., “Mllms are deeply affected by modality bias,” arXiv preprint arXiv:2505.18657, 2025.
- [59] Z. Wu, T. Liu, L. Luo, Z. Zhong, J. Chen, H. Xiao, C. Hou, H. Lou, Y. Chen, R. Yang et al., “Mars: An instance-aware, modular and realistic simulator for autonomous driving,” in CAAI International Conference on Artificial Intelligence. Springer, 2023, pp. 3–15.
- [60] Y. Fu, R. Wang, Y. Fu, D. P. Paudel, X. Huang, and L. Van Gool, “Objectrelator: Enabling cross-view object relation understanding in ego-centric and exo-centric videos,” ICCV, 2025.
- [61] Y. Li, Q. Ma, R. Yang, H. Li, M. Ma, B. Ren, N. Popovic, N. Sebe, E. Konukoglu, T. Gevers et al., “Scenesplat: Gaussian splatting-based scene understanding with vision-language pretraining,” in ICCV, 2025.
- [62] T. Br¨odermann, C. Sakaridis, Y. Fu, and L. Van Gool, “Cafuser: Condition-aware multimodal fusion for robust semantic perception of driving scenes,” IEEE Robotics and Automation Letters, 2025.
- [63] M. Ma, Q. Ma, Y. Li, J. Cheng, R. Yang, B. Ren, N. Popovic, M. Wei, N. Sebe, L. Van Gool et al., “Scenesplat++: A large dataset and comprehensive benchmark for language gaussian splatting,” arXiv

- preprint arXiv:2506.08710, 2025.
- [64] Y. Lyu, X. Zheng, J. Zhou, and L. Wang, “Unibind: Llm-augmented unified and balanced representation space to bind them all,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 26752–26762.
- [65] X. Zheng, Z. Weng, Y. Lyu, L. Jiang, H. Xue, B. Ren, D. Paudel, N. Sebe, L. Van Gool, and X. Hu, “Retrieval augmented generation and understanding in vision: A survey and new outlook,” arXiv preprint arXiv:2503.18016, 2025.
- [66] J. Zhou, X. Zheng, Y. Lyu, and L. Wang, “Eventbind: Learning a unified representation to bind them all for event-based open-world understanding,” in European Conference on Computer Vision. Springer, 2024, pp. 477–494.
- [67] X. Zheng, Y. Lyu, and L. Wang, “Learning modalityagnostic representation for semantic segmentation from any modalities,” in European Conference on Computer Vision. Springer, 2024, pp. 146–165.
- [68] Y. Lyu, X. Zheng, D. Kim, and L. Wang, “Omnibind: Teach to build unequal-scale modality interaction for omni-bind of all,” arXiv preprint arXiv:2405.16108, 2024.
- [69] Z. Dongfang, X. Zheng, Z. Weng, Y. Lyu, D. P. Paudel, L. Van Gool, K. Yang, and X. Hu, “Are multimodal large language models ready for omnidirectional spatial reasoning?” arXiv preprint arXiv:2505.11907, 2025.
- [70] X. Zhang, Z. Ye, and X. Zheng, “Towards omnidirectional reasoning with 360-r1: A dataset, benchmark, and grpo-based method,” arXiv preprint arXiv:2505.14197, 2025.
- [71] G. Zhou, P. Qiu, C. Chen, J. Wang, Z. Yang, J. Xu, and M. Qiu, “Reinforced mllm: A survey on rl-based reasoning in multimodal large language models,” arXiv preprint arXiv:2504.21277, 2025.
- [72] C. Wang, T. Zhang, R. Hong, and J. Huang, “A short survey on small reasoning models: Training, inference, applications and research directions,” arXiv preprint arXiv:2504.09100, 2025.
- [73] Z. Ke, F. Jiao, Y. Ming, X.-P. Nguyen, A. Xu, D. X. Long, M. Li, C. Qin, P. Wang, S. Savarese et al., “A survey of frontiers in llm reasoning: Inference scaling, learning to reason, and agentic systems,” arXiv preprint arXiv:2504.09037, 2025.
- [74] J. Bi, S. Liang, X. Zhou, P. Liu, J. Guo, Y. Tang, L. Song, C. Huang, G. Sun, J. He et al., “Why reasoning matters? a survey of advancements in multimodal reasoning (v1),” arXiv preprint arXiv:2504.03151, 2025.
- [75] Z. Chen, S. Wang, Z. Tan, X. Fu, Z. Lei, P. Wang, H. Liu, C. Shen, and J. Li, “A survey of scaling in large language model reasoning,” arXiv preprint arXiv:2504.02181, 2025.
- [76] Q. Chen, L. Qin, J. Liu, D. Peng, J. Guan, P. Wang, M. Hu, Y. Zhou, T. Gao, and W. Che, “Towards reasoning era: A survey of long chain-of-thought for reasoning large language models,” arXiv preprint arXiv:2503.09567, 2025.

- [77] A. Forootani, “A survey on mathematical reasoning and optimization with large language models,” arXiv preprint arXiv:2503.17726, 2025.
- [78] R. Wang, H. Wang, B. Xue, J. Pang, S. Liu, Y. Chen, J. Qiu, D. F. Wong, H. Ji, and K.-F. Wong, “Harnessing the reasoning economy: A survey of efficient reasoning for large language models,” arXiv preprint arXiv:2503.24377, 2025.
- [79] Y. Liu, J. Wu, Y. He, R. Gong, J. Xia, L. Li, H. Gao, H. Chen, B. Bi, J. Zhang et al., “Efficient inference for large reasoning models: A survey,” arXiv preprint arXiv:2503.23077, 2025.
- [80] X. Qu, Y. Li, Z. Su, W. Sun, J. Yan, D. Liu, G. Cui, D. Liu, S. Liang, J. He et al., “A survey of efficient reasoning for large reasoning models: Language, multimodality, and beyond,” arXiv preprint arXiv:2503.21614, 2025.
- [81] Z. Lin, Y. Gao, X. Zhao, Y. Yang, and J. Sang, “Mind with eyes: from language reasoning to multimodal reasoning,” arXiv preprint arXiv:2503.18071, 2025.
- [82] Y. Sui, Y.-N. Chuang, G. Wang, J. Zhang, T. Zhang, J. Yuan, H. Liu, A. Wen, S. Zhong, H. Chen et al., “Stop overthinking: A survey on efficient reasoning for large language models,” arXiv preprint arXiv:2503.16419, 2025.
- [83] Y. Wang, S. Wu, Y. Zhang, S. Yan, Z. Liu, J. Luo, and H. Fei, “Multimodal chain-of-thought reasoning: A comprehensive survey,” arXiv preprint arXiv:2503.12605, 2025.
- [84] D. Bandyopadhyay, S. Bhattacharjee, and A. Ekbal, “Thinking machines: A survey of llm based reasoning strategies,” arXiv preprint arXiv:2503.10814, 2025.
- [85] X. Li, Z. Cai, S. Wang, K. Yu, and F. Chen, “A survey on enhancing causal reasoning ability of large language models,” in Pacific-Asia Conference on Knowledge Discovery and Data Mining. Springer, 2025, pp. 399–416.
- [86] Y. Yan, J. Su, J. He, F. Fu, X. Zheng, Y. Lyu, K. Wang, S. Wang, Q. Wen, and X. Hu, “A survey of mathematical reasoning in the era of multimodal large language model: Benchmark, method & challenges,” arXiv preprint arXiv:2412.11936, 2024.
- [87] D. Yang, T. Liu, D. Zhang, A. Simoulin, X. Liu, Y. Cao, Z. Teng, X. Qian, G. Yang, J. Luo et al., “Code to think, think to code: A survey on code-enhanced reasoning and reasoning-driven code intelligence in llms,” arXiv preprint arXiv:2502.19411, 2025.
- [88] Z.-Z. Li, D. Zhang, M.-L. Zhang, J. Zhang, Z. Liu, Y. Yao, H. Xu, J. Zheng, P.-J. Wang, X. Chen et al., “From system 1 to system 2: A survey of reasoning large language models,” arXiv preprint arXiv:2502.17419, 2025.
- [89] F. Cheng, H. Li, F. Liu, R. van Rooij, K. Zhang, and Z. Lin, “Empowering llms with logical reasoning: A comprehensive survey,” arXiv preprint arXiv:2502.15652, 2025.
- [90] G. Srivastava, S. Cao, and X. Wang, “Towards reasoning ability of small language models,” arXiv preprint arXiv:2502.11569, 2025.

- [91] F. Xu, Q. Hao, Z. Zong, J. Wang, Y. Zhang, J. Wang, X. Lan, J. Gong, T. Ouyang, F. Meng et al., “Towards large reasoning models: A survey of reinforced reasoning with large language models,” arXiv preprint arXiv:2501.09686, 2025.
- [92] Y. Wang, W. Chen, X. Han, X. Lin, H. Zhao, Y. Liu, B. Zhai, J. Yuan, Q. You, and H. Yang, “Exploring the reasoning abilities of multimodal large language models (mllms): A comprehensive survey on emerging trends in multimodal reasoning,” arXiv preprint arXiv:2401.06805, 2024.
- [93] J. Wang, Y. Ming, Z. Shi, V. Vineet, X. Wang, Y. Li, and N. Joshi, “Is a picture worth a thousand words? delving into spatial reasoning for vision language models,” in The Thirty-Eighth Annual Conference on Neural Information Processing Systems, 2024.
- [94] Y. Shu, B. Ren, Z. Xiong, D. P. Paudel, L. Van Gool, B. Demir, N. Sebe, and P. Rota, “Earthmind: Towards multi-granular and multi-sensor earth observation with large multimodal models,” arXiv preprint arXiv:2506.01667, 2025.
- [95] C. Li, C. Zhang, H. Zhou, N. Collier, A. Korhonen, and I. Vulic, “Topviewrs: Vision-language models as top-view spatial reasoners,” in Proceedings of the Conference on Empirical Methods in Natural Language Processing, 2024.
- [96] M. Jia, Z. Qi, S. Zhang, W. Zhang, X. Yu, J. He, H. Wang, and L. Yi, “Omnispatial: Towards comprehensive spatial reasoning benchmark for vision language models,” arXiv preprint arXiv:2506.03135, 2025.
- [97] R. Wu and D. Guo, “Do large language models have spatial cognitive abilities?” ACM Transactions on Intelligent Systems and Technology, 2025.
- [98] Y. Liao, X. Liu, C. Wang, Z. Liu, Y. Zhang, and Y. Zhu, “Reasoning paths with reference objects elicit quantitative spatial reasoning in large vision-language models,” in Proceedings of the Conference on Empirical Methods in Natural Language Processing, 2024.
- [99] Y. Zhang, J. Han, L. Wang, L. Guibas, and S. Xie, “Spatial understanding from videos: Structured prompts meet simulation data,” arXiv preprint arXiv:2506.03642, 2025.
- [100] Z. Zhou et al., “Image-of-thought prompting for visual reasoning refinement in multimodal large language models,” arXiv preprint arXiv:2405.13872, 2024.
- [101] F. Zhu, H. Wang, Y. Xie, J. Gu, T. Ding, J. Yang, and H. Jiang, “Struct2d: A perception-guided framework for spatial reasoning in large multimodal models,” arXiv preprint arXiv:2506.04220, 2025.
- [102] P. Y. Lee, J. Je, C. Park, M. A. Uy, L. Guibas, and M. Sung, “Perspective-aware reasoning in visionlanguage models via mental imagery simulation,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2025.
- [103] Y. Meng et al., “I know about “up”! enhancing spatial reasoning in visual language models through 3d reconstruction,” arXiv preprint arXiv:2407.14133, 2024.
- [104] R. Liu, R. Wu, B. Van Hoorick, P. Tokmakov, S. Za-

- kharov, and C. Vondrick, “Zero-1-to-3: Zero-shot one image to 3d object,” in Proceedings of the IEEE/CVF international conference on computer vision, 2023, pp. 9298–9309.
- [105] L. Marsili, L. Sforza, L. Barsellotti, N. Amoroso, and A. Monaco, “Visual agentic ai for spatial reasoning with a dynamic api,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025.
- [106] S. Ravi, M. Chen, A. Sen, S. Shivakumar, E. Sklar, and E. Santos, “Out of sight, not out of context? egocentric spatial reasoning in vlms across disjoint frames,” arXiv preprint arXiv:2505.24257, 2024.
- [107] J. Feng, Y. Du, T. Liu, S. Guo, Y. Lin, and Y. Li, “Citygpt: Empowering urban spatial cognition of large language models,” arXiv preprint arXiv:2406.13948, 2024.
- [108] H. Li, J. Chen, Z. Wei, S. Huang, T. Hui, J. Gao, X. Wei, and S. Liu, “Llava-st: A multimodal large language model for fine-grained spatial-temporal understanding,” arXiv preprint arXiv:2501.08282, 2025.
- [109] P. Wu, Y. Liu, M. Liu, and J. Shen, “St-think: How multimodal large language models reason about 4d worlds from ego-centric videos,” arXiv preprint arXiv:2503.12542, 2025.
- [110] A. Ray, J. Duan, R. Tan, D. Bashkirova, R. Hendrix, K. Ehsani, A. Kembhavi, B. A. Plummer, R. Krishna, K.-H. Zeng et al., “Sat: Spatial aptitude training for multimodal language models,” arXiv preprint arXiv:2412.07755, vol. 3, 2024.
- [111] O. Ogezi et al., “Spare: Enhancing spatial reasoning in vision-language models with synthetic data,” arXiv preprint arXiv:2504.20648, 2025.
- [112] K. Tang et al., “Sparkle: Mastering basic spatial capabilities in vision language models elicits generalization to spatial reasoning,” arXiv preprint arXiv:2410.16162, 2025.
- [113] J. Ko et al., “St-vlm: Kinematic instruction tuning for spatio-temporal reasoning in vision-language models,” arXiv preprint arXiv:2503.19355, 2025.
- [114] C. Li, W. Wu, H. Zhang, Y. Xia, S. Mao, L. Dong,

I. Vuli´c, and F. Wei, “Imagine while reasoning in space: Multimodal visualization-of-thought,” arXiv preprint arXiv:2501.07542, 2025.

- [115] X. Liang, X. Guo, Z. Jin, W. Pan, P. Shang, D. Cai, B. Lin, and J. Ye, “Enhancing spatial reasoning through visual and textual thinking,” arXiv preprint arXiv:2507.20529, 2025.
- [116] Z. Pan et al., “Metaspatial: Reinforcing 3d spatial reasoning in vlms for the metaverse,” arXiv preprint arXiv:2503.18470, 2025.
- [117] Z. Liao, Q. Xie, Y. Zhang, Z. Kong, H. Lu, Z. Yang, and Z. Deng, “Improved visual-spatial reasoning via r1-zero-like training,” arXiv preprint arXiv:2504.00883, 2025.
- [118] Y. Wang et al., “M2-reasoning: Empowering mllms with unified general and spatial reasoning,” arXiv preprint arXiv:2507.08306, 2025.

- [119] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark et al., “Learning transferable visual models from natural language supervision,” in International conference on machine learning. PmLR, 2021, pp. 8748–8763.
- [120] M. Cherti, R. Beaumont, R. Wightman, M. Wortsman, G. Ilharco, C. Gordon, C. Schuhmann, L. Schmidt, and J. Jitsev, “Reproducible scaling laws for contrastive language-image learning,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2023, pp. 2818–2829.
- [121] Q. Sun, Y. Fang, L. Wu, X. Wang, and Y. Cao, “Evaclip: Improved training techniques for clip at scale,” arXiv preprint arXiv:2303.15389, 2023.
- [122] M. Tschannen, A. Gritsenko, X. Wang, M. F. Naeem,

I. Alabdulmohsin, N. Parthasarathy, T. Evans, L. Beyer, Y. Xia, B. Mustafa et al., “Siglip 2: Multilingual visionlanguage encoders with improved semantic understanding, localization, and dense features,” arXiv preprint arXiv:2502.14786, 2025.

- [123] X. Zhai, B. Mustafa, A. Kolesnikov, and L. Beyer, “Sigmoid loss for language image pre-training,” in Proceedings of the IEEE/CVF international conference on computer vision, 2023, pp. 11975–11986.
- [124] W. Ma, L. Ye, C. de Melo, A. L. Yuille, and J. Chen, “Spatialllm: A compound 3d-informed design towards spatially-intelligent large multimodal models,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2025.
- [125] Z. Zhang, X. Li, Z. Xu, W. Peng, Z. Zhou, M. Shi, and S. Huang, “Mpdrive: Improving spatial understanding with marker-based prompt learning for autonomous driving,” arXiv preprint arXiv:2504.00379, 2025.
- [126] K. Ranasinghe, S. N. Shukla, O. Poursaeed, M. S. Ryoo, and T.-Y. Lin, “Learning to localize objects improves spatial reasoning in visual-llms,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 12977–12987.
- [127] W. Cai, I. Ponomarenko, J. Yuan, X. Li, W. Yang, H. Dong, and B. Zhao, “Spatialbot: Precise spatial understanding with vision language models,” arXiv preprint arXiv:2406.13642, 2024.
- [128] Y. Liu, M. Ma, X. Yu, P. Ding, H. Zhao, M. Sun, S. Huang, and D. Wang, “Ssr: Enhancing depth perception in vision-language models via rationale-guided spatial reasoning,” arXiv preprint arXiv:2505.12448, 2025.
- [129] H. Zheng, B. Tian, M. Wu, Z. Tang, K. Nahrstedt, and A. G. Schwing, “Spatio-temporal llm: Reasoning about environments and actions,” arXiv preprint arXiv:2507.05258, 2025.
- [130] E. Daxberger, N. Wenzel, D. Griffiths, H. Gang, J. Lazarow, G. Kohavi, K. Kang, M. Eichner, Y. Yang, A. Dehghan, and P. Grasch, “Mm-spatial: Exploring 3d spatial understanding in multimodal llms,” arXiv preprint arXiv:2503.13111, 2025.
- [131] S. Tong, E. Brown, P. Wu, S. Woo, M. Middepogu, S. C. Akula, J. Yang, S. Yang, A. Iyer, X. Pan, Z. Wang,

- R. Fergus, Y. LeCun, and S. Xie, “Cambrian-1: A fully open, vision-centric exploration of multimodal llms,” in Advances in Neural Information Processing Systems (NeurIPS), 2024.
- [132] R. Rajabi et al., “Towards grounded visual spatial reasoning in multi-modal vision language models,” in ICLR Workshop, 2024.
- [133] W. Zhang, Y. Huang, Y. Xu, J. Huang, H. Zhi, S. Ren, W. Xu, and J. Zhang, “Why do mllms struggle with spatial understanding? a systematic analysis from data to architecture,” arXiv preprint arXiv:2509.02359, 2025.
- [134] T.-Y. Wu, S.-Y. Huang, and Y.-C. F. Wang, “Dataefficient 3d visual grounding via order-aware referring,” in 2025 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV). IEEE, 2025, pp. 3107– 3117.
- [135] Z. Guo, Y. Tang, R. Zhang, D. Wang, Z. Wang, B. Zhao, and X. Li, “Viewrefer: Grasp the multi-view knowledge for 3d visual grounding,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 15372–15383.
- [136] Z. Yuan, J. Ren, C.-M. Feng, H. Zhao, S. Cui, and Z. Li, “Visual programming for zero-shot open-vocabulary 3d visual grounding,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 20623–20633.
- [137] B. M. Ocal,¨ M. Tatarchenko, S. Karao˘glu, and T. Gevers, “Sceneteller: Language-to-3d scene generation,” in European Conference on Computer Vision. Springer, 2024, pp. 362–378.
- [138] Q. Wu, D. Iliash, D. Ritchie, M. Savva, and A. X. Chang, “Diorama: Unleashing zero-shot single-view 3d scene modeling,” arXiv preprint arXiv:2411.19492, 2024.
- [139] J. Wen, Y. Zhu, J. Li, Z. Tang, C. Shen, and F. Feng, “Dexvla: Vision-language model with plug-in diffusion expert for general robot control,” arXiv preprint arXiv:2502.05855, 2025.
- [140] R. Zheng, Y. Liang, S. Huang, J. Gao, H. Daum´e III, A. Kolobov, F. Huang, and J. Yang, “Tracevla: Visual trace prompting enhances spatial-temporal awareness for generalist robotic policies,” arXiv preprint arXiv:2412.10345, 2024.
- [141] L. Zhao, D. Cai, L. Sheng, and D. Xu, “3dvgtransformer: Relation modeling for visual grounding on point clouds,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2021, pp. 2928–2937.
- [142] J. Huang, B. Jia, Y. Wang, Z. Zhu, X. Linghu, Q. Li, S.-C. Zhu, and S. Huang, “Unveiling the mist over 3d vision-language understanding: Object-centric evaluation with chain-of-analysis,” arXiv preprint arXiv:2503.22420, 2025.
- [143] C. Zhu, T. Wang, W. Zhang, K. Chen, and X. Liu, “Scanreason: Empowering 3d visual grounding with reasoning capabilities,” in European Conference on Computer Vision. Springer, 2024, pp. 151–168.
- [144] Y.-H. Yang, L. Piccinelli, M. Segu, S. Li, R. Huang,

- Y. Fu, M. Pollefeys, H. Blum, and Z. Bauer, “3dmood: Lifting 2d to 3d for monocular open-set object detection,” ICCV, 2025.
- [145] A. Kirillov, E. Mintun, N. Ravi, H. Mao, C. Rolland, L. Gustafson, T. Xiao, S. Whitehead, A. C. Berg, W.Y. Lo et al., “Segment anything,” in Proceedings of the IEEE/CVF international conference on computer vision, 2023, pp. 4015–4026.
- [146] Z. Qi, Y. Fang, Z. Sun, X. Wu, T. Wu, J. Wang, D. Lin, and H. Zhao, “Gpt4point: A unified framework for point-language understanding and generation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 26417– 26427.
- [147] J. Huang, S. Yong, X. Ma, X. Linghu, P. Li, Y. Wang, Q. Li, S.-C. Zhu, B. Jia, and S. Huang, “An embodied generalist agent in 3d world,” arXiv preprint arXiv:2311.12871, 2023.
- [148] R. Fu, J. Liu, X. Chen, Y. Nie, and W. Xiong, “Scene-llm: Extending language model for 3d visual understanding and reasoning,” arXiv preprint arXiv:2403.11401, 2024.
- [149] N. Zantout, H. Zhang, P. Kachana, J. Qiu, J. Zhang, and W. Wang, “Sort3d: Spatial object-centric reasoning toolbox for zero-shot 3d grounding using large language models,” arXiv preprint arXiv:2504.18684, 2025.
- [150] Z. Wang, H. Huang, Y. Zhao, Z. Zhang, and Z. Zhao, “Chat-3d: Data-efficiently tuning large language model for universal dialogue of 3d scenes,” arXiv preprint arXiv:2308.08769, 2023.
- [151] H. Huang, Y. Chen, Z. Wang, R. Huang, R. Xu, T. Wang, L. Liu, X. Cheng, Y. Zhao, J. Pang et al., “Chat-scene: Bridging 3d scene and large language models with object identifiers,” arXiv preprint arXiv:2312.08168, 2023.
- [152] Y. Hong, H. Zhen, P. Chen, S. Zheng, Y. Du, Z. Chen, and C. Gan, “3d-llm: Injecting the 3d world into large language models,” Advances in Neural Information Processing Systems, vol. 36, pp. 20482–20494, 2023.
- [153] J. Deng, T. He, L. Jiang, T. Wang, F. Dayoub, and I. Reid, “3d-llava: Towards generalist 3d lmms with omni superpoint transformer,” arXiv preprint arXiv:2501.01163, 2025.
- [154] B. Kerbl, G. Kopanas, T. Leimk¨uhler, and G. Drettakis, “3d gaussian splatting for real-time radiance field rendering.” 2023.
- [155] Q. Ma, Y. Li, B. Ren, N. Sebe, E. Konukoglu, T. Gevers, L. Van Gool, and D. P. Paudel, “A large-scale dataset of gaussian splats and their self-supervised pretraining,” in 3DV. IEEE, 2025, pp. 145–155.
- [156] J. Li, D. Li, S. Savarese, and S. Hoi, “Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models,” in International conference on machine learning. PMLR, 2023, pp. 19730–19742.
- [157] H. Xiong, Y. Zhuge, J. Zhu, L. Zhang, and H. Lu, “3ur-llm: An end-to-end multimodal large language model for 3d scene understanding,” arXiv preprint

- arXiv:2501.07819, 2025.
- [158] Z. Li, C. Zhang, X. Wang, R. Ren, Y. Xu, R. Ma, X. Liu, and R. Wei, “3dmit: 3d multi-modal instruction tuning for scene understanding,” in 2024 IEEE International Conference on Multimedia and Expo Workshops (ICMEW). IEEE, 2024, pp. 1–5.
- [159] H. Yu, W. Li, S. Wang, J. Chen, and J. Zhu, “Inst3d-lmm: Instance-aware 3d scene understanding with multi-modal instruction tuning,” arXiv preprint arXiv:2503.00513, 2025.
- [160] A. Thai, S. Peng, K. Genova, L. Guibas, and T. Funkhouser, “Splattalk: 3d vqa with gaussian splatting,” arXiv preprint arXiv:2503.06271, 2025.
- [161] Z. Qi, Z. Zhang, Y. Fang, J. Wang, and H. Zhao, “Gpt4scene: Understand 3d scenes from videos with vision-language models,” arXiv preprint arXiv:2501.01428, 2025.
- [162] L. Ling, C.-H. Lin, T.-Y. Lin, Y. Ding, Y. Zeng, Y. Sheng, Y. Ge, M.-Y. Liu, A. Bera, and Z. Li, “Scenethesis: A language and vision agentic framework for 3d scene generation,” arXiv preprint arXiv:2505.02836, 2025.
- [163] Y. Yang, F.-Y. Sun, L. Weihs, E. VanderBilt, A. Herrasti, W. Han, J. Wu, N. Haber, R. Krishna, L. Liu et al., “Holodeck: Language guided generation of 3d embodied ai environments,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 16227–16237.
- [164] A. ¸Celen, G. Han, K. Schindler, L. Van Gool, I. Armeni, A. Obukhov, and X. Wang, “I-design: Personalized llm interior designer,” in European Conference on Computer Vision. Springer, 2025, pp. 217–234.
- [165] Q. He, K. Lin, S. Chen, A. Hu, and Q. Jin, “Thinkprogram-rectify: 3d situated reasoning with large language models,” arXiv preprint arXiv:2404.14705, 2024.
- [166] L. Jiang, R. Ji, and L. Zhang, “Sdf-3dgan: A 3d object generative method based on implicit signed distance function,” arXiv preprint arXiv:2303.06821, 2023.
- [167] L. Jiang, J. Lin, K. Chen, W. Ge, X. Yang, Y. Jiang, Y. Lyu, X. Zheng, Y. Li, and Y. Chen, “Dimer: Disentangled mesh reconstruction model,” arXiv preprint arXiv:2504.17670, 2025.
- [168] L. Jiang, H. Li, and L. Wang, “A general framework to boost 3d gs initialization for text-to-3d generation by lexical richness,” in Proceedings of the 32nd ACM International Conference on Multimedia, 2024, pp. 6803– 6812.
- [169] L. Jiang, X. Zheng, Y. Lyu, J. Zhou, and L. Wang, “Brightdreamer: Generic 3d gaussian generative framework for fast text-to-3d synthesis,” arXiv preprint arXiv:2403.11273, 2024.
- [170] T. Hua, L. Jiang, Y.-C. Chen, and W. Zhao, “Sat2city: 3d city generation from a single satellite image with cascaded latent diffusion,” arXiv preprint arXiv:2507.04403, 2025.
- [171] Y. Sasazawa and Y. Sogawa, “Layout generation agents with large language models,” arXiv preprint arXiv:2405.08037, 2024.

- [172] Y. Yang, J. Lu, Z. Zhao, Z. Luo, J. J. Yu, V. Sanchez, and F. Zheng, “Llplace: The 3d indoor scene layout generation and editing via large language model,” arXiv preprint arXiv:2406.03866, 2024.
- [173] C. Wang, H. Zhong, M. Chai, M. He, D. Chen, and J. Liao, “Chat2layout: Interactive 3d furniture layout with a multimodal llm,” arXiv preprint arXiv:2407.21333, 2024.
- [174] Cursor, “Cursor: The ai-powered code editor,” 2023, accessed: 2025-06-19. [Online]. Available: https:// www.cursor.so/
- [175] GitHub, “Github copilot,” 2021, accessed: 2025-06-19. [Online]. Available: https://copilot.github.com/
- [176] V. Kumaran, J. Rowe, B. Mott, and J. Lester, “Scenecraft: automating interactive narrative scene generation in digital games with large language models,” in Proceedings of the AAAI Conference on Artificial Intelligence and Interactive Digital Entertainment, vol. 19, no. 1, 2023, pp. 86–96.
- [177] H. I. I. Tam, H. I. D. Pun, A. T. Wang, A. X. Chang, and M. Savva, “Scenemotifcoder: Exampledriven visual program learning for generating 3d object arrangements,” arXiv preprint arXiv:2408.02211, 2024.
- [178] S. Wang, C. Chen, X. Le, Q. Xu, L. Xu, Y. Zhang, and J. Yang, “Cad-gpt: Synthesising cad construction sequence with spatial reasoning-enhanced multimodal llms,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 39, no. 8, 2025, pp. 7880–7888.
- [179] X. Wang, J. Zheng, Y. Hu, H. Zhu, Q. Yu, and Z. Zhou, “From 2d cad drawings to 3d parametric models: A vision-language approach,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 39, no. 8, 2025, pp. 7961–7969.
- [180] J. Li, W. Ma, X. Li, Y. Lou, G. Zhou, and X. Zhou, “Cad-llama: leveraging large language models for computer-aided design parametric 3d model generation,” in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 18563–18573.
- [181] R. K. Jones, P. Guerrero, N. J. Mitra, and D. Ritchie, “Shapelib: designing a library of procedural 3d shape abstractions with large language models,” arXiv preprint arXiv:2502.08884, 2025.
- [182] Z. Zhou, Y. Zhu, M. Zhu, J. Wen, N. Liu, Z. Xu, W. Meng, R. Cheng, Y. Peng, C. Shen et al., “Chatvla: Unified multimodal understanding and robot control with vision-language-action model,” arXiv preprint arXiv:2502.14420, 2025.
- [183] S. K. Ramakrishnan, A. Gokaslan, E. Wijmans, O. Maksymets, A. Clegg, J. Turner, E. Undersander, W. Galuba, A. Westbury, A. X. Chang et al., “Habitat-matterport 3d dataset (hm3d): 1000 largescale 3d environments for embodied ai,” arXiv preprint arXiv:2109.08238, 2021.
- [184] H. R. Walke, K. Black, T. Z. Zhao, Q. Vuong, C. Zheng, P. Hansen-Estruch, A. W. He, V. Myers, M. J. Kim, M. Du et al., “Bridgedata v2: A dataset for robot learning at scale,” in Conference on Robot Learning. PMLR, 2023, pp. 1723–1736.

- [185] X. Zheng, C. Liao, Z. Weng, K. Lei, Z. Dongfang, H. He, Y. Lyu, L. Jiang, L. Qi, L. Chen et al., “Panorama: The rise of omnidirectional vision in the embodied ai era,” arXiv preprint arXiv:2509.12989, 2025.
- [186] H. Gardner and T. Hatch, “Educational implications of the theory of multiple intelligences,” Educational researcher, vol. 18, no. 8, pp. 4–10, 1989.
- [187] A. Kamath, J. Hessel, and K.-W. Chang, “What’s” up” with vision-language models? investigating their struggle with spatial reasoning,” arXiv preprint arXiv:2310.19785, 2023.
- [188] K. Li, Q. Xu, T. Qian, Y. Fu, Y. Jiao, and X. Wang, “Clivis: Unleashing cognitive map through linguisticvisual synergy for embodied visual reasoning,” arXiv preprint arXiv:2506.17629, 2025.
- [189] M. J. Kim, K. Pertsch, S. Karamcheti, T. Xiao, A. Balakrishna, S. Nair, R. Rafailov, E. P. Foster, P. R. Sanketi, Q. Vuong et al., “Openvla: An open-source visionlanguage-action model,” in 8th Annual Conference on Robot Learning, 2024.
- [190] K. Black, N. Brown, D. Driess, A. Esmail, M. Equi, C. Finn, N. Fusai, L. Groom, K. Hausman,

B. Ichter et al., “π0: A vision-language-action flow model for general robot control,” arXiv preprint arXiv:2410.24164, 2024.

- [191] L. X. Shi, B. Ichter, M. Equi, L. Ke, K. Pertsch, Q. Vuong, J. Tanner, A. Walling, H. Wang, N. Fusai et al., “Hi robot: Open-ended instruction following with hierarchical vision-language-action models,” arXiv preprint arXiv:2502.19417, 2025.
- [192] C. Li, J. Wen, Y. Peng, Y. Peng, F. Feng, and Y. Zhu, “Pointvla: Injecting the 3d world into vision-languageaction models,” arXiv preprint arXiv:2503.07511, 2025.
- [193] D. Qu, H. Song, Q. Chen, Y. Yao, X. Ye, Y. Ding, Z. Wang, J. Gu, B. Zhao, D. Wang et al., “Spatialvla: Exploring spatial representations for visual-languageaction model,” arXiv preprint arXiv:2501.15830, 2025.
- [194] P. Li, Y. Chen, H. Wu, X. Ma, X. Wu, Y. Huang, L. Wang, T. Kong, and T. Tan, “Bridgevla: Inputoutput alignment for efficient 3d manipulation learning with vision-language models,” arXiv preprint arXiv:2506.07961, 2025.
- [195] B. Zitkovich, T. Yu, S. Xu, P. Xu, T. Xiao, F. Xia, J. Wu, P. Wohlhart, S. Welker, A. Wahid et al., “Rt-2: Vision-language-action models transfer web knowledge to robotic control,” in Conference on Robot Learning. PMLR, 2023, pp. 2165–2183.
- [196] J. Yang, R. Tan, Q. Wu, R. Zheng, B. Peng, Y. Liang, Y. Gu, M. Cai, S. Ye, J. Jang et al., “Magma: A foundation model for multimodal ai agents,” in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 14203–14214.
- [197] M. Zawalski, W. Chen, K. Pertsch, O. Mees, C. Finn, and S. Levine, “Robotic control via embodied chain-ofthought reasoning,” in 8th Annual Conference on Robot Learning, 2024.
- [198] J. Li, Y. Zhu, Z. Tang, J. Wen, M. Zhu, X. Liu, C. Li,

- R. Cheng, Y. Peng, and F. Feng, “Improving visionlanguage-action models via chain-of-affordance,” arXiv preprint arXiv:2412.20451, 2024.
- [199] S. Nasiriany, S. Kirmani, T. Ding, L. Smith, Y. Zhu, D. Driess, D. Sadigh, and T. Xiao, “Rt-affordance: Affordances are versatile intermediate representations for robot manipulation,” in 1st Workshop on X-Embodiment Robot Learning, 2024.
- [200] S. Karamcheti, S. Nair, A. Balakrishna, P. Liang, T. Kollar, and D. Sadigh, “Prismatic VLMs: Investigating the Design Space of Visually-Conditioned Language Models.”
- [201] L. Beyer, A. Steiner, A. S. Pinto, A. Kolesnikov, X. Wang, D. Salz, M. Neumann, I. Alabdulmohsin, M. Tschannen, E. Bugliarello et al., “Paligemma: A versatile 3b vlm for transfer,” arXiv preprint arXiv:2407.07726, 2024.
- [202] M. Abdin, J. Aneja, H. Behl, S. Bubeck, R. Eldan, S. Gunasekar, M. Harrison, R. J. Hewett, M. Javaheripi, P. Kauffmann et al., “Phi-4 technical report,” arXiv preprint arXiv:2412.08905, 2024.
- [203] P. Wang, S. Bai, S. Tan, S. Wang, Z. Fan, J. Bai, K. Chen, X. Liu, J. Wang, W. Ge et al., “Qwen2vl: Enhancing vision-language model’s perception of the world at any resolution,” arXiv preprint arXiv:2409.12191, 2024.
- [204] S. K. Ramakrishnan, E. Wijmans, P. Kraehenbuehl, and V. Koltun, “Does spatial cognition emerge in frontier models?” arXiv preprint arXiv:2410.06468, 2024.
- [205] Y. Zhang, Z. Ma, J. Li, Y. Qiao, Z. Wang, J. Chai, Q. Wu, M. Bansal, and P. Kordjamshidi, “Visionand-language navigation today and tomorrow: A survey in the era of foundation models,” arXiv preprint arXiv:2407.07035, 2024.
- [206] Q. Gu, A. Kuwajerwala, S. Morin, K. M. Jatavallabhula, B. Sen, A. Agarwal, C. Rivera, W. Paul, K. Ellis, R. Chellappa et al., “Conceptgraphs: Open-vocabulary 3d scene graphs for perception and planning,” in 2024 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2024, pp. 5021–5028.
- [207] D. Zheng, S. Huang, L. Zhao, Y. Zhong, and L. Wang, “Towards learning a generalist model for embodied navigation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 13624–13634.
- [208] Z. Wang and G. H. Lee, “g3d-lf: Generalizable 3dlanguage feature fields for embodied tasks,” arXiv preprint arXiv:2411.17030, 2024.
- [209] Y. Yang, H. Yang, J. Zhou, P. Chen, H. Zhang, Y. Du, and C. Gan, “3d-mem: 3d scene memory for embodied exploration and reasoning,” in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 17294–17303.
- [210] W. Yuan, J. Duan, V. Blukis, W. Pumacay, R. Krishna, A. Murali, A. Mousavian, and D. Fox, “Robopoint: A vision-language model for spatial affordance prediction for robotics,” arXiv preprint arXiv:2406.10721, 2024.
- [211] Y. Liu, D. Chi, S. Wu, Z. Zhang, Y. Hu, L. Zhang,

- Y. Zhang, S. Wu, T. Cao, G. Huang et al., “Spatialcot: Advancing spatial reasoning through coordinate alignment and chain-of-thought for embodied task planning,” arXiv preprint arXiv:2501.10074, 2025.
- [212] S. Zhang, Y. Qiao, Q. Wang, L. Guo, Z. Wei, and J. Liu, “Flexvln: Flexible adaptation for diverse vision-and-language navigation tasks,” arXiv preprint arXiv:2503.13966, 2025.
- [213] L. Zhang, X. Hao, Y. Tang, H. Fu, X. Zheng, P. Wang, Z. Wang, W. Ding, and S. Zhang, “Nava3:ˆ Understanding any instruction, navigating anywhere, finding anything,” arXiv preprint arXiv:2508.04598, 2025.
- [214] L. Zhong, C. Gao, Z. Ding, Y. Liao, H. Ma, S. Zhang, X. Zhou, and S. Liu, “Topv-nav: Unlocking the topview spatial reasoning potential of mllm for zero-shot object navigation,” arXiv preprint arXiv:2411.16425,

- 2024.

[215] L. Ling and B. Qianqian, “Endowing embodied agents with spatial reasoning capabilities for vision-andlanguage navigation,” arXiv preprint arXiv:2504.08806,

- 2025.

- [216] S. Tan, M. Ge, D. Guo, H. Liu, and F. Sun, “Knowledgebased embodied question answering,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 45, no. 10, pp. 11948–11960, 2023.
- [217] B. Zhao, Z. Wang, J. Fang, C. Gao, F. Man, J. Cui, X. Wang, X. Chen, Y. Li, and W. Zhu, “Embodied-r: Collaborative framework for activating embodied spatial reasoning in foundation models via reinforcement learning,” arXiv preprint arXiv:2504.12680, 2025.
- [218] A. Das, S. Datta, G. Gkioxari, S. Lee, D. Parikh, and D. Batra, “Embodied question answering,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2018, pp. 1–10.
- [219] Y. Tang, S. Zhang, X. Hao, P. Wang, J. Wu, Z. Wang, and S. Zhang, “Affordgrasp: In-context affordance reasoning for open-vocabulary task-oriented grasping in clutter,” arXiv preprint arXiv:2503.00778, 2025.
- [220] X. Guo, H. Hu, C. Song, J. Chen, Z. Zhao, Y. Fu, B. Guan, and Z. Liu, “Unidiffgrasp: A unified framework integrating vlm reasoning and vlm-guided part diffusion for open-vocabulary constrained grasping with dual arms,” arXiv preprint arXiv:2505.06832, 2025.
- [221] H. Zhi, P. Chen, S. Zhou, Y. Dong, Q. Wu, L. Han, and M. Tan, “3dflowaction: Learning cross-embodiment manipulation from 3d flow world model,” arXiv preprint arXiv:2506.06199, 2025.
- [222] A. Hurst, A. Lerer, A. P. Goucher, A. Perelman, A. Ramesh, A. Clark, A. Ostrow, A. Welihinda, A. Hayes, A. Radford et al., “Gpt-4o system card,” arXiv preprint arXiv:2410.21276, 2024.
- [223] B. Liu, Y. Dong, Y. Wang, Y. Rao, Y. Tang, W.C. Ma, and R. Krishna, “Coarse correspondence elicit 3d spacetime understanding in multimodal language model,” arXiv preprint arXiv:2408.00754, 2024.
- [224] A. Team, H. Zhu, Y. Wang, J. Zhou, W. Chang, Y. Zhou, Z. Li, J. Chen, C. Shen, J. Pang, and T. He, “Aether: Geometric-aware unified world modeling,”

- arXiv preprint arXiv:2503.18945, 2025.
- [225] Z. Cheng, J. Hu, Z. Liu, C. Si, W. Li, and S. Gong, “V-star: Benchmarking video-llms on video spatiotemporal reasoning,” arXiv preprint arXiv:2503.11495, 2025.
- [226] C. Tang, W. Yu, G. Sun, X. Chen, T. Tan, W. Li, J. Zhang, L. Lu, Z. Ma, Y. Wang et al., “Can large language models understand spatial audio?” arXiv preprint arXiv:2406.07914, 2024.
- [227] H. Yun, Y. Yu, W. Yang, K. Lee, and G. Kim, “Panoavqa: Grounded audio-visual question answering on 360deg videos,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2021, pp. 2031–2041.
- [228] C. Wen, T. Guo, S. Zhao, W. Zou, and X. Li, “Sari: Structured audio reasoning via curriculum-guided reinforcement learning,” arXiv preprint arXiv:2504.15900, 2025.
- [229] S. Chowdhury, S. Nag, S. Dasgupta, J. Chen, M. Elhoseiny, R. Gao, and D. Manocha, “Meerkat: Audio-visual large language model for grounding in space and time,” 2024.
- [230] Z. Xing, X. Hu, C.-W. Fu, W. Wang, J. Dai, and P.-A. Heng, “EchoInk-R1: Exploring audio-visual reasoning in multimodal LLMs via reinforcement learning,” arXiv preprint arXiv:2505.04623, 2025.
- [231] R. Krishna, Y. Zhu, O. Groth, J. Johnson, K. Hata, J. Kravitz, S. Chen, Y. Kalantidis, L.-J. Li, D. A. Shamma et al., “Visual genome: Connecting language and vision using crowdsourced dense image annotations,” International journal of computer vision, vol. 123, pp. 32–73, 2017.
- [232] K. Yang, O. Russakovsky, and J. Deng, “Spatialsense: An adversarially crowdsourced benchmark for spatial relation recognition,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2019, pp. 2051–2060.
- [233] J. Lei, L. Yu, T. L. Berg, and M. Bansal, “Tvqa+: Spatio-temporal grounding for video question answering,” arXiv preprint arXiv:1904.11574, 2019.
- [234] Y.-C. Su, S. Changpinyo, X. Chen, S. Thoppay, C.-J. Hsieh, L. Shapira, R. Soricut, H. Adam, M. Brown, M.-H. Yang et al., “2.5 d visual relationship detection,” Computer Vision and Image Understanding, vol. 224, p. 103557, 2022.
- [235] L. Parcalabescu, M. Cafagna, L. Muradjan, A. Frank,

I. Calixto, and A. Gatt, “Valse: A task-independent benchmark for vision and language models centered on linguistic phenomena,” arXiv preprint arXiv:2112.07566, 2021.

- [236] F. Liu, G. Emerson, and N. Collier, “Visual spatial reasoning,” Transactions of the Association for Computational Linguistics, vol. 11, pp. 635–651, 2023.
- [237] A. Kamath, J. Hessel, and K.-W. Chang, “What’s“ up” with vision-language models? investigating their struggle with spatial reasoning,” arXiv preprint arXiv:2310.19785, 2023.
- [238] X. Guo, R. Zhang, Y. Duan, Y. He, C. Zhang, S. Liu,

- and L. Chen, “Drivemllm: A benchmark for spatial understanding with multimodal large language models in autonomous driving,” arXiv preprint arXiv:2411.13112, 2024.
- [239] J. Wang, Y. Ming, Z. Shi, V. Vineet, X. Wang, S. Li, and N. Joshi, “Is a picture worth a thousand words? delving into spatial reasoning for vision language models,” Advances in Neural Information Processing Systems, vol. 37, pp. 75392–75421, 2024.
- [240] N. Rajabi and J. Kosecka, “Gsr-bench: A benchmark for grounded spatial reasoning evaluation via multimodal llms,” arXiv preprint arXiv:2406.13246, 2024.
- [241] X. Wang, W. Ma, T. Zhang, C. M. de Melo, J. Chen, and A. Yuille, “Pulsecheck457: A diagnostic benchmark for comprehensive spatial reasoning of large multimodal models,” arXiv preprint arXiv:2502.08636, 2025.
- [242] H. Liu, C. Li, Y. Li, and Y. J. Lee, “Improved baselines with visual instruction tuning,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2024, pp. 26296–26306.
- [243] J. Achiam, S. Adler, S. Agarwal, L. Ahmad, I. Akkaya, F. L. Aleman, D. Almeida, J. Altenschmidt, S. Altman, S. Anadkat et al., “Gpt-4 technical report,” arXiv preprint arXiv:2303.08774, 2023.
- [244] X. Fu, Y. Hu, B. Li, Y. Feng, H. Wang, X. Lin, D. Roth, N. A. Smith, W.-C. Ma, and R. Krishna, “Blink: Multimodal large language models can see but not perceive,” in European Conference on Computer Vision. Springer, 2024, pp. 148–166.
- [245] W. Dai, J. Li, D. Li, A. Tiong, J. Zhao, W. Wang, B. Li, P. N. Fung, and S. Hoi, “Instructblip: Towards general-purpose vision-language models with instruction tuning,” Advances in neural information processing systems, vol. 36, pp. 49250–49267, 2023.
- [246] G. Team, R. Anil, S. Borgeaud, J.-B. Alayrac, J. Yu, R. Soricut, J. Schalkwyk, A. M. Dai, A. Hauth, K. Millican et al., “Gemini: a family of highly capable multimodal models,” arXiv preprint arXiv:2312.11805, 2023.
- [247] B. Li, Y. Zhang, D. Guo, R. Zhang, F. Li, H. Zhang, K. Zhang, P. Zhang, Y. Li, Z. Liu et al., “Llavaonevision: Easy visual task transfer,” arXiv preprint

- arXiv:2408.03326, 2024.

[248] P. Wang, S. Bai, S. Tan, S. Wang, Z. Fan, J. Bai, K. Chen, X. Liu, J. Wang, W. Ge et al., “Qwen2vl: Enhancing vision-language model’s perception of the world at any resolution,” arXiv preprint

- arXiv:2409.12191, 2024.

- [249] J. Bai, S. Bai, Y. Chu, Z. Cui, K. Dang, X. Deng, Y. Fan, W. Ge, Y. Han, F. Huang et al., “Qwen technical report,” arXiv preprint arXiv:2309.16609, 2023.
- [250] Q. Ye, H. Xu, J. Ye, M. Yan, A. Hu, H. Liu, Q. Qian, J. Zhang, and F. Huang, “mplug-owl2: Revolutionizing multi-modal large language model with modality collaboration,” in Proceedings of the ieee/cvf conference on computer vision and pattern recognition, 2024, pp. 13040–13051.
- [251] G. Team, P. Georgiev, V. I. Lei, R. Burnell, L. Bai, A. Gulati, G. Tanzer, D. Vincent, Z. Pan, S. Wang

- et al., “Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context,” arXiv preprint arXiv:2403.05530, 2024.
- [252] M. Xu, M. Wu, Y. Zhao, J. C. L. Li, and W. Ou, “Llavaspacesgg: Visual instruct tuning for open-vocabulary scene graph generation with enhanced spatial relations,” in 2025 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV). IEEE, 2025, pp. 6362– 6372.
- [253] Z. Wang, Y. Dong, F. Luo, M. Ruan, Z. Cheng, C. Chen, P. Li, and Y. Liu, “How do multimodal large language models handle complex multimodal reasoning? placing them in an extensible escape game,” arXiv preprint arXiv:2503.10042, 2025.
- [254] C. Plizzari, A. Tonioni, Y. Xian, A. Kulshrestha, and F. Tombari, “Omnia de egotempo: Benchmarking temporal understanding of multi-modal llms in egocentric videos,” in CVPR, 2025.
- [255] Y. Li, Y. Fu, T. Qian, Q. Xu, S. Dai, D. P. Paudel, L. Van Gool, and X. Wang, “Egocross: Benchmarking multimodal large language models for cross-domain egocentric video question answering,” arXiv preprint arXiv:2508.10729, 2025.
- [256] Z. Wen, Y. Wang, C. Liao, B. Yang, J. Li, W. Liu, H. He, B. Feng, X. Liu, Y. Lyu et al., “Ai for service: Proactive assistance with ai glasses,” arXiv preprint arXiv:2510.14359, 2025.
- [257] X. Zhang, T. Fu, and X. Zheng, “Omnidirectional spatial modeling from correlated panoramas,” arXiv preprint arXiv:2509.02164, 2025.
- [258] C. Liao, K. Lei, X. Zheng, J. Moon, Z. Wang, Y. Wang, D. P. Paudel, L. Van Gool, and X. Hu, “Benchmarking multi-modal semantic segmentation under sensor failures: Missing and noisy modality robustness,” in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 1576–1586.
- [259] X. Zheng, Y. Lyu, J. Zhou, and L. Wang, “Centering the value of every modality: Towards efficient and resilient modality-agnostic semantic segmentation,” in European Conference on Computer Vision. Springer, 2024, pp. 192–212.
- [260] X. Zheng, Y. Lyu, L. Jiang, D. P. Paudel, L. Van Gool, and X. Hu, “Reducing unimodal bias in multi-modal semantic segmentation with multi-scale functional entropy regularization,” arXiv preprint arXiv:2505.06635, 2025.

