## CADEvolve: Creating Realistic CAD via Program Evolution

Maksim Elistratov1 Marina Barannikov2 Gregory Ivanov1 Valentin Khrulkov4

Anton Konushin1 Andrey Kuznetsov34 Dmitrii Zhemchuzhnikov1†

# arXiv:2602.16317v1[cs.GR]18Feb2026

1Lomonosov Moscow State University; 2Universit´e Paris Dauphine;

- 3 Innopolis University;
- 4 FusionBrain Lab, AXXX

### Abstract

Computer-Aided Design (CAD) delivers rapid, editable modeling for engineering and manufacturing. Recent AI progress now makes full automation feasible for various CAD tasks. However, progress is bottlenecked by data: public corpora mostly contain sketch–extrude sequences, lack complex operations, multi-operation composition and design intent, and thus hinder effective fine-tuning. Attempts to bypass this with frozen VLMs often yield simple or invalid programs due to limited 3D grounding in current foundation models. We present CADEvolve, an evolutionbased pipeline and dataset that starts from simple primitives and, via VLM-guided edits and validations, incrementally grows CAD programs toward industrial-grade complexity. The result is ≈8k complex parts expressed as executable CadQuery parametric generators. After multi-stage postprocessing and augmentation, we obtain a unified dataset of ≈1.3m scripts paired with rendered geometry and exercising the full CadQuery operation set. A VLM fine-tuned on CADEvolve achieves state-of-the-art results on the Image2CAD task across the DeepCAD, Fusion 360, and MCB benchmarks. Code, dataset, and the SOTA model are available at GitHub, Hugging Face dataset, and Hugging Face model.

### 1. Introduction

Computer-Aided Design (CAD) has transformed engineering by enabling precise, parametric modeling and rapid iteration, yet the next leap—AI automation—remains data-

†Corresponding author: zhemchuzhnikovds@my.msu.ru

###### (a) Representation (c) Propose (d) Retrieve + Synthesis

| | |
|---|---|
| |parent<br><br>name|

| | |
|---|---|
|parent K<br><br>abstract<br><br>detailed<br><br>name<br><br>k|def|

name

1) parents 2) text-embed. nearest neighbors

P = { name = "box_prism", abstract = "rectangular prism", detailed = Box LxWxH..." , code = param2cq(z)→S, parents = ... }

abstract

detailed

code

corpus

(f) Select & growth

(e) Multi-stage validation

pa

t 1

###### na

[Figure 1]

L W H

abstract

ab

detailed

de

CQ obj

p2cq(z):

[Figure 2]

param2cq(z)→S

generation

Execution errors

[Figure 3]

[Figure 4]

proposes

Execution check

(b) Seed pool

repair

Shape is not a single solid

[Figure 5]

corpus

[Figure 6]

[Figure 7]

[Figure 8]

candidate

Geometry validity

repair

x k

name

[Figure 9]

abstract

[Figure 10]

Compare with description

detailed

[Figure 11]

[Figure 12]

[Figure 13]

parents

[Figure 14]

Vis.-text agreement

repair

Figure 1. CADEvolve overview. (a) Representation of a shape tuple; (b) seed pool of 46 hand-written generators; (c) VLM proposals conditioned on sampled parents; (d) retrieval-augmented code synthesis; (e) staged validation (execution check, geometry validity, visual–text agreement) with targeted repair; (f) selection and growth of the accepted pool.

limited. In practice, programs are built as sequences of 2D sketches and 3D operations [57, 58]; prior corpora and systems represent these either as command tokens [5, 18, 58] or as concise, executable Python (e.g., CADQUERY [2]) [8, 42, 55]. However, public CAD sequence corpora effectively collapse to sketch–extrude pipelinese.g., Fusion 360 Gallery, DeepCAD, and CAD-Recodewhile richer operations (revolve, loft, sweep, fillet, chamfer, shell, local patterns) are absent from released program histories. This limits the learning of multi-operation composition and design intent. While Seek-CAD [27] and RLCAD [62] report support for additional operations (e.g., revolve, chamfer , fillet), neither releases multi-operation construction histories, so the community still lacks an open corpus that systematically exercises a broad operator set.

On the geometry side, there are many datasets of real CAD shapes—ABC [21], ShapeNet [4], MCB [20]—but they do not provide CAD sequences. DeepCAD [58] and Fusion 360 Gallery [57] contain parts with user-authored histories, yet their usable sequence splits are predominantly prismatic and sketch–extrude only. To scale data volume, CAD-Recode [42] introduced a rule-based generator, but it still produces only sketches and extrusions; extending it to other operations would require brittle constraint systems to avoid geometric collisions and still does not guarantee richer topology or design intent. A parallel line of work uses frozen VLMs to synthesize code from prompts—e.g., 3DPreMise [66], CADCodeVerify [1], and Seek-CAD [27]but single-pass prompting typically yields simple shapes with a narrow operator set, even with retrieval augmentation or multi-stage validation.

Early in this project, we found that single-pass VLMs struggle to reconstruct industrial-grade CAD programs: they tend to saturate on extruded prisms and fail to chain heterogeneous operations reliably. Recent work suggests a way around this: pair an LLM that proposes code with automated evaluators and evolve candidates via selection, yielding results beyond one-shot capabilities—e.g., the AlphaEvolve coding agent reports new state-of-the-art algorithmic solutions [36]. Motivated by this evidence, we introduce CADEvolve: an evolutionary propose–execute– filter pipeline for CAD data generation. Starting from 46 hand-written CADQUERY primitives, a VLM (GPT-5-mini in our experiments [37]) repeatedly edits and extends parent programs; each candidate must compile to a solid and pass geometry checks. Successful programs become parents for the next round. This pipeline yields 7,945 valid parametric generators (maps from shape parameters → CADQUERY solids), i.e., each generator represents a class of parts. Unlike EvoCAD [38], which applies evolutionary search at inference time, we use evolution only as an offline data generator.

From these generators we next build a training corpus. First, we parse each generator and sample parameters, producing ∼ 8×105 runnable programs with paired geometry. Supervised fine-tuning on this set produces diverse, valid multi-operation code, but geometric fidelity to targets remains insufficient. To strengthen supervision without altering external datasets, we run the imperfect model on meshonly corpora (ABC, ShapeNet) and collect its predicted programs and corresponding shapes as augmentation; we do not modify ABC and ShapeNet themselves—only harvest our model’s outputs on their meshes. We then introduce canonicalization: unify code templates and symbol names, normalize pose and scale, and binarize numeric parameters. This produces ∼ 3.5×105 canonicalized scripts after filtering. Finally, we diversify early-tree structure by mixing a balanced, canonicalized subset derived from CAD-

Recode primitives with our canonicalized scripts, yielding a ∼ 7×105 training set. Models trained on this set improve substantially; RL fine-tuning with geometry-derived rewards closes the remaining gap and achieves state-ofthe-art Image2CAD performance on DeepCAD, Fusion 360 Gallery, and MCB.

For clarity of nomenclature, we denote the three tiers as CADEvolve-3L: CADEvolve-G (7,945 parametric generators), CADEvolve-P (∼ 8×105 executable programs sampled from generators, with paired geometry), and CADEvolve-C (∼ 8×105 canonicalized normalized binarized scripts used for training). Unless stated otherwise, all models are trained on CADEvolve-C. Crucially, CADEvolve uses evolution only for offline dataset synthesis; inference is a single-model decode.

##### Contributions.

- • CADEvolve (pipeline). An offline propose–execute– filter evolutionary pipeline that generates complex multioperation CADQUERY programs, enabling realistic synthetic data when open corpora are small.
- • CADEvolve-3L (dataset). A three-tier corpus—G (parametric generators), P (executable CADQUERY scripts with arbitrary code style), and C (canonicalized programs for training)—that is the first open CAD sequence dataset covering the full CADQUERY operator set with executable multi-operation histories.
- • CADEvolve-M (policy). A vision–language model finetuned on CADEvolve-C for the Image2CAD task. It supports the full CADQUERY operator set and achieves stateof-the-art reconstruction performance on DeepCAD, Fusion 360 Gallery, and MCB under a fixed architecture and RL recipe.

### 2. Related Work

CAD generation. Research on CAD synthesis spans three target representations: CSG trees, B-reps, and program/sequence models. CSG focuses on primitive boolean composition and struggles to capture the variety and detail of engineered parts [9–11, 17, 35, 39, 47, 63, 64]. B-rep generators reason over faces, edges and topology but tend to be brittle and harder to edit [13, 16, 23, 26, 28, 30, 31, 45, 51, 54, 61]. Sequence models — either command tokens (sketch / extrude / boolean) or concise, executable Python (e.g., CADQUERY) — best match parametric workflows and preserve editability [3, 6, 8, 15, 18, 19, 24, 25, 33, 34, 40, 53, 58–60, 65, 69]. Yet the open sequence corpora the community trains on are largely sketch–extrude: Fusion 360 Gallery and DeepCAD provide mostly prismatic histories, while CAD-Recode scales volume via a rule-based extrusion generator [42, 57, 58]. Mesh-only sets (e.g., ABC, ShapeNet, MCB) supply geometry but no editable histories, limiting program-level supervision [4, 20, 21]. CADEvolve

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

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

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

- Figure 2. Examples of generated parts. A diverse gallery of accepted CADEvolve-G outputs spanning extrude, revolve, loft, sweep, shell, fillet, chamfer, booleans, and local patterns. Colors are arbitrary.

targets this gap by releasing multi-operation, executable histories.

Case-based CAD program generation. To inject domain priors at inference, case-based methods retrieve related designs and structure prompts. Seek-CAD applies RAG and a self-refinement loop for parametric modeling [27]. In parallel, several works standardize code formats to improve executability and editing — OpenECAD, InstructGraph — and CAD-Llama’s SPCC adds a hierarchical semantic layout [25, 50, 67]. These strategies curb hallucinations but ultimately inherit the limits of available corpora: while Seek-CAD and RLCAD report support for revolve, fillet and chamfer, corresponding multi-operation histories are not publicly released [27, 62]. CADEvolve provides such histories explicitly, together with a canonicalized layer tuned for training.

Evolutionary methods. LLM-driven evolution — propose, evaluate, and select — has enabled problems beyond single-pass generation: FunSearch, LLaMEA / LLaMEAHPO, and AlphaEvolve demonstrate competitive algorithm discovery via tight validator loops [36, 41, 48, 49]. In CAD, EvoCAD deploys evolution at inference, maintaining a population ranked by text similarity and refined via crossover and mutation with a self-debug filter [38]. CADEvolve relocates evolution to the data stage: starting from hand-written CADQUERY primitives, a VLM proposes edits; only programs that compile to solids and pass geometry checks survive. This offline propose–execute–filter process yields parametric generators and large batches of executable multi-operation scripts suitable for pretraining; no population search is used at test time.

Image2CAD. Recent work reconstructs CAD sequences directly from visual inputs. Early single-view approaches such as CSGNet predict compact CSG programs from a single raster image of a synthetic shape [44]. Subsequent work has framed Image2CAD as a reinforcement-learning problem, optimizing token-level policies against rendering-

or geometry-based rewards and scaling to richer CAD corpora [62, 68]. More recent models — CADCrafter, CADCoder, and cadrille — consume multi-view grids or isometric renderings of CAD parts and emit parametric programs that better match industrial design workflows [5, 8, 22]. In this work, we adopt Image2CAD as the simplest controlled setting to validate datasets: fixed multi-view renderings fully specify the target geometry without requiring pershape textual descriptions (as in Text2CAD) or additional point-cloud encoders (as in PC2CAD), making architectural confounders minimal and comparisons more transparent.

RL for CAD reconstruction. Two lines exist. (i) Environment-level RL optimizes command sequences with geometry-based rewards without LLMs (REINFORCE in CSGNet; DQN for orthographic drawings; RLCAD’s Brep gym) [44, 62, 68]. (ii) LLM post-training aligns code generators using verifiable signals: DPO with codecheckers or visual feedback (CADCrafter, CADFusion) and GRPO-family objectives with geometry-aware rewards (CAD-Coder, cadrille, GACO-CAD) [5, 12, 22, 52, 55]. Our contributions are orthogonal to these RL recipes: we re-use standard GRPO-style objectives and baselines, but introduce a multi-operation, executable, canonicalized corpus that benefits both SFT-only and RLVR pipelines, providing the operator diversity and code regularity missing from prior open datasets.

### 3. Dataset Generation and Processing

#### 3.1. Evolutionary Synthesis of Parametric Generators (CADEvolve-G)

Representation. We represent a shape as a tuple P = {name, abstract, detailed, code, parents} (Fig. 3), where name specifies the part name in the snake case, abstract and detailed are abstract (concise description) and detailed descriptions, respectively, the code is the mapping param2cq : z →S, a self–contained CADQUERY function mapping a collection of semantic parameters z to a single watertight solid S. For different values of parameters, we

obtain different solids, e.g., gears of varying radii. The textual fields capture design intent; parents records lineage for inheritance.

Seed pool. We begin with an initial corpus of 46 hand-written generators that collectively cover extrude, revolve, loft, sweep, shell, fillet, chamfer, booleans, and local patterns/arrays (e.g., gears, wedges, prisms, torus segments). This pool anchors operator breadth and parameterization styles.

Propose–Execute–Filter loop. The initial pool D0 is seeded by the previously discussed manually constructed programs. Given the current accepted pool Dt we perform the following steps:

- 1. Parent sampling. Randomly sample K parents from Dt to encourage recombination across operations.
- 2. Child metadata proposal. gpt5-mini is asked to propose k children, each providing name, abstract, detailed, and the list of parents it inherits from. Proposals must imply a single solid body and avoid an assembly of repeating solids and encourage more complex geometry than their parents.
- 3. Code synthesis with retrieval. For each child, we retrieve a small set of nearest neighbors by embedding the detailed description and union it with the sampled parents’ code. gpt5-mini then produces a monolithic, parametric CADQUERY function param2cq, explicitly exposing design–meaningful parameters with default values that are used to call the generator and valid the execution and the output shape.
- 4. Staged validation and self-repair.

- • Execution check: param2cq must compile and run on defaults, returning exactly one solid.
- • Geometry validity: the result passes strict CAD integrity tests.

[Figure 177]

- Figure 3. Representation of a shape from CADEvolve-G. The representation consists of descriptive textual fields, a Python code that maps geometric parameters to a 3D shape S, and the list of parents from which S has evolved via the CADEvolve algorithm.

• Visual–text agreement: we render a seven-view montage (one isometric + six orthographic projections) and ask the VLM to verify that the rendered geometry matches the child’s abstract and detailed descriptions. If any stage fails, the model is prompted to issue a targeted fix.

5. Selection and growth. Only children that pass all verification stages are admitted to Dt+1; we store their metadata and lineage. The loop repeats until a fixed budget is met or novelty saturates (see Supplementary A).

This procedure yields 7,945 validated parametric generators CADEvolve-G. An overview of the pipeline is shown in Fig. 1, and representative evolutionary lineages are visualized in Supplementary B.

#### 3.2. Sampling and Parsing of Generators (CADEvolve-P)

Goal. Given this set of generators, we extend it as follows. For each parametric generator, we extract its set of parameters with default values z and search for a small diverse set of variations of this initial vector z1,...,zN (we use N = 15) that (i) produce valid solids and (ii) cover distinct regions of the design space param2cq.

Quality–diversity objective. We define our fitness objective as follows. Given an arbitrary parameter vector as an input to the generator, we build the corresponding shape and compute the penalty with two terms. (1) Validity/fit: If CAD checks fail (not exactly one watertight solid), assign a large penalty. Otherwise, add small penalties if the longest side falls outside the [60,200] unit range or if any face of the axis-aligned bounding box exits the cube [−100,100]3.

(2) Novelty: Compare the candidate to an archive of accepted samples. If it lies closer than a distance threshold ε to any archived point, add a non-negative penalty that grows as the gap to ε increases. When both parts are zero, the sample is valid and novel.

Search. Given the non-differentiable nature of the task, we use the well-known black box optimization approach CMA-ES [14] with default parameters as the initial vector to find unique parameter vectors producing valid shapes. We iterate until we collect N accepted samples per generator (we use N=15) or a compute budget is reached, yielding a compact set of valid, diverse instances for each generator.

From generators to concrete programs. For each accepted parameter vector z and its generator param2cq, we emit a deterministic CADQUERY script, termed CADEvolve-P, via single-run tracing and slicing:

1. Header. Insert minimal imports (import cadquery as cq; import math only if used).

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

###### A  B 

[Figure 183]

[Figure 184]

D 

[Figure 185]

[Figure 186]

C 

- Figure 4. From generator to concrete program. Given a (A) parametric generator param2cq and a sampled parameter vector z, we (B) bind parameters (param i = z["param i"]), then execute the generator once to resolve conditionals and loops, skipping untaken branches and retaining only geometry-affecting CADQUERY operations; (C) apply code–level augmentation; and (D) emit a flat, deterministic script with minimal imports and a unified output (result = ...). The script reproduces the shape exactly while exposing its construction history.

- 2. Parameter materialization. Emit one bind per argument in the exact order expected by param2cq: param i = z["param i"].

- 3. Trace & slice. Execute param2cq(z) under a tracer; record only constructive CADQUERY operations that contribute to the final solid (sketch ops; extrude/revolve/loft/sweep; fillet/chamfer/shell; booleans; patterns). Drop checks, logging, and other non-geometric statements.
- 4. Control-flow resolution. Keep only executed branches and realized loop bodies; no residual if/try remain in the output.
- 5. Standardize the output variable name. Emit a flat, side-effect–free sequence with stable formatting, minimal imports, and a unified tail line result = .... The resulting script contains parameter binds from z; it

deterministically reproduces the shape and makes the construction history explicit. These scripts form CADEvolveP and are used downstream for training and evaluation (Fig. 4). We target 10 scripts per generator; this stage yields 74,918 scripts in total.

#### 3.3. Direct Image2CAD training and code augmentation

Starting from ≈ 75k generator–script pairs, we observed severe template collapse: within each generator, scripts

shared the same identifiers and operation sequence, differing only in argument values. A small Qwen2-VL-2B model trained on this data learned spurious correlations between geometry and a fixed code skeleton and reproduced training templates; results were unusable, so we omit metrics. To break this bias while preserving geometry, we applied code–level augmentation: for every script we asked a compact LLM (gpt-5-mini) to produce up to 10 semantically equivalent rewrites (different structure, same solid). We over-generated and kept only validated, executable variants, yielding in total 744,780 scripts. During this pass we also pruned non-contributing operations (e.g., dead sketches, unused workplanes, no-op fillets), keeping each script minimal yet functionally identical.

#### 3.4. Bootstrapping with Image2CAD pretraining and mesh2CAD distillation

- Round 1 (Image2CAD on rewrites). We trained Qwen2VL-2B on Image2CAD using the 744,780 rewritten scripts as targets (multi-view renders as inputs). The model produced diverse, valid code but shapes were far from the targets; quantitative results are deferred to Sec. 4.
- Round 2 (adding ABC/ShapeNet predictions). To expand coverage, we used the Round-1 model to predict CADQUERY programs for meshes from ABC and ShapeNet, then filtered to keep only valid scripts whose boundingbox max-extent lies in [60,200]. This produced 875,632 ABC scripts and 119,437 ShapeNet scripts. Combined with our rewrites, the training set totaled 1,739,849 scripts (≈ 1.74M). We retrained Qwen2-VL-2B on the enlarged corpus; metric gains were incremental and still below state-ofthe-art, motivating further dataset improvements (Sec. 4).

#### 3.5. CADEvolve-C: Canonicalization & Normalization

Our initial corpus, while valid, was overly sophisticated and exhibited wide scale variation (max extent in [60,200]). We therefore enforce a unified format, size, and numeric grid so the learner focuses on construction logic rather than incidental syntax or scale.

- 1. Unification. Remove residual non-geometric Python; keep only geometry-affecting CADQUERY calls. Reemit as a flat, macro-like sequence with stable temporaries (wp1, wp2, ...) and minimal imports.
- 2. Centering. Build the solid, compute its AABB center, and inject a deterministic translation so the final object is centered at (0,0,0).
- 3. Extent normalization. Apply a uniform scale so the bounding box longest side equals a fixed target (200 units), yielding a canonical size (roughly within [−100,100]3).

Table 1. CAD sequence generation conditioned on multi-view images. When CADEvolve-M (RL1) is trained on the same RL set as cadrille , it achieves lower CD and higher IoU than cadrille on all three datasets, at the cost of higher IR, reflecting more frequent use of complex operations that are more collision-prone than sketch–extrude pipelines. Adding MCB to the RL training pool trades a small drop in CD/IoU on DeepCAD and Fusion360—bringing them close to cadrille’s levels—for a substantial improvement on MCB.

DeepCAD Fusion360 MCB

Method CD↓ IoU↑ IR↓ CD↓ IoU↑ IR↓ CD↓ IoU↑ IR↓ cadrille SFT 0.19 86.5 1.6 0.20 77.3 3.4 1.16 40.4 14.3 cadrille RL 0.17 92.2 0.1 0.17 84.6 0.1 0.87 47.6 2.5

CADEvolve-P pre-aug (SFT) 7.31 37.2 17.3 9.15 29.7 19.1 13.19 17.3 25.3 CADEvolve-P post-aug (SFT) 4.93 42.9 14.3 7.54 33.1 16.1 10.98 20.2 22.3

CADEvolve-C small (SFT) 3.40 49.6 19.1 9.25 35.1 26.1 8.45 28.1 43.1 CADEvolve-C middle (SFT) 0.57 70.1 23.6 0.68 59.1 25.2 2.09 39.1 37.7 CADEvolve-C big (SFT) 0.67 72.1 19.0 0.26 71.1 18.2 1.71 42.0 32.0

CADEvolve-C big (RL1) 0.15 92.6 0.2 0.16 87.2 0.5 0.62 51.4 2.3 CADEvolve-C big (RL2) 0.16 91.1 0.1 0.16 84.0 0.2 0.52 55.2 0.4

- 4. Binarization. Quantize all numeric literals after scaling: zero-out tiny epsilons and round remaining values to integers. This removes floating-point noise and constrains the parameter space to a consistent grid.

As illustrated in Fig. 4, we convert traced generators into canonical, centered, and uniformly scaled CADQUERY programs before training.

Collision-aware pruning after CADEvolve-C. Canonicalization occasionally induced geometric collisions. We re-validated all scripts post-transform and kept only valid ones, yielding 1,002,002 programs in total: 69,201 from CADEvolve, 813,378 ABC predictions, and 119,312 ShapeNet predictions. Note that canonicalization was applied to CADEvolve-P before code-style augmentation; the rewrite pass mainly enabled diverse ABC/ShapeNet predictions but they are not part of this final canonicalized layer since the output of canonization of different rewrites for one shape is identical.

Length-based filtering and truncation. We split the original dataset into two groups based on script length: 849,558 scripts had fewer than 3k characters, while 152,444 scripts exceeded 3k characters. For the longer scripts, we applied truncation to 3k characters and successfully truncated 151,892 scripts. We then passed these truncated scripts through the canonicalization pipeline again (excluding the unification stage, since it had already been applied). As a result, we obtained 129,961 valid scripts. After a final deduplication step, these corresponded to 111,742 unique scripts. Therefore, the final number of scripts shorter than 3k characters is 961,300.

Post-canonization baseline. Training QWEN2-VL-2B on this set improved results markedly but remained below SOTA (see Sec. 4). Error analysis pointed to limited sketch diversity.

Sketch-diversity augmentation. To inject variability in early sketches, we targeted CADEvolve scripts whose canonicalized first primitives hit the normalization bound (e.g., axis-aligned box with max extent 200 or cylinder with diameter/height 200). We replaced the base primitive with a script from CADRecode (known for diverse sketches). This contributed 963,096 additional scripts.

Mesh generation. Using the scripts from the lengthbased filtering step and the sketch-diversity augmentation step, we generated STL meshes for the resulting set of scripts. Due to rendering and geometry validation failures, STL files were successfully produced for only 1,382,928 scripts.

Rotational augmentation. Using rotational augmentation (see Supplementary C), we obtained an additional 1,337,553 scripts. The final supervised fine-tuning (SFT) training set thus comprises 2,720,481 scripts.

Dataset characteristics. Key dataset characteristics, including operation occurrence statistics, sequence length, and geometric face count, are reported in Supplementary D.

сadrille

DeepCAD Fusion 360 MCB

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

[Figure 193]

[Figure 194]

[Figure 195]

CADEvolve

[Figure 196]

[Figure 197]

[Figure 198]

Target

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

- Figure 5. Qualitative comparisons across datasets. Columns: DeepCAD, Fusion360, MCB. Rows (top→bottom): the cadrille baseline, our CADEvolve–C big (RL2) prediction, and the target render. This panel best illustrates the advantages of the CADEvolve dataset: targets include parts built via revolve, sweep, loft, face selectors, and complex hole patterns that cannot be well approximated by the sketch- extrude-boolean schemes used in CAD-Recode and many other datasets, where cadrille typically fails but CADEvolve-M closely reconstructs the input shape.

### 4. CADEvolve-M: Program-Generating Policy

#### 4.1. Problem and Training Pipeline

Problem. To validate the dataset, we adopt the simplest controllable setting: Image2CAD. Given a fixed multi-view render of a shape, the model emits a CADQUERY program that compiles to a solid matching the target. Unlike Text2CAD, this task does not need textual descriptions, which we do not have for scripts from augmentations; unlike PC2CAD, it does not require training a point-cloud encoder. We feed the multi-view image grid directly into the VLM’s built-in visual encoder, introducing no extra image backbone, adapters, or pretraining beyond the base model.

Experiment setup. Experiments on CADEvolve-P used 7 views (6 orthographic + 1 iso); final experiments on CADEvolve-C use 8 canonical views: six orthographic projections (±X,±Y,±Z) and two isometric views. Shapes are rigid-aligned and lie in [−100,100]3. For each orthographic view we render a 238×238 image of that box and encode depth along the view axis via intensity. To keep axis directions consistent, −Z, +Y , +X images are horizontally mirrored. The eight images are concatenated into a 2×4 grid fed to the model. Unlike the 4-iso setup in cadrille [22], the 6-ortho + 1–2 iso layout sharpens cues for fillets and chamfers. We use QWEN2-VL-2B as the vi-

sion–language backbone, prompting it with the multi-view grid and decoding CADQUERY tokens. For experiments on CADEvolve-P we normalized the target shape before visualizations and put its center and maximum extent in the prompt which is not needed for CADEvolve-C where all the shapes are normalized. An overview of the end-to-end Image2CAD training pipeline (SFT→RL) is shown in Fig. 6.

#### 4.2. Datasets

SFT corpus. We did experiments on different steps of CADEvolve-P and CADEvolve-C processing.

RL & Evaluation data. We ground both RL fine-tuning and evaluation on three public corpora: DeepCAD, Fusion360, and MCB. To ensure coverage and reduce category bias, we re-split MCB so that the test set spans all ISO categories. For RL, both runs use the cadrille RL train set (selected parts from the DeepCAD and Fusion360 train splits); in the second run, we additionally include the MCB training split while keeping its test split fixed

#### 4.3. Metrics

All meshes are rigid-aligned and normalized to [0,1]3. We report (i) Chamfer Distance (CD) on 8,192 vs 8,192 points (scaled by 103), (ii) volumetric IoU (%), and (iii) Invalid Rate (IR) — fraction of generations that fail to compile or

- 1. SFT
- 2. RL

[Figure 214]

###### CADEvolve

###### pairs (stl, script)

Mesh

[Figure 215]

[Figure 216]

[Figure 217]

visual encoder

textual decoder

Dr.CPPO loss

8-view Image2CAD image

###### Prediction

[Figure 218]

Predicted mesh

Reward (IoU)

[Figure 219]

[Figure 220]

Real CAD shapes

stl

Mesh Mesh

- Figure 6. Image2CAD training pipeline. Multi-view inputs (7 views for CADEvolve-P; 8 canonical views for CADEvolveC/RL) are fed to the VLM’s built-in visual encoder; a textual decoder emits CADQUERY code. We first run SFT on paired (render, script) data, then apply online RL (Dr.GRPO + CPPO) with an IoU-based reward and invalidity penalties, using the predicted mesh for feedback.

yield a non-watertight/degenerate solid. To reduce invalidity bias we report median CD.

#### 4.4. Training

SFT. We perform two epochs of supervised fine-tuning in each experiment. Objective: token-level cross-entropy on code conditioned on views.

RL fine-tuning. We adopt the same online RL training and reward as cadrille, i.e., the GRPO objective with Dr.GRPO and CPPO variants (Dr. CPPO) [29, 32, 43], and a programmatic reward that combines IoU (scaled to emphasize accuracy) with a penalty for invalid generations (non-compiling or non-watertight) [22]. The reward is r = 10 · IoU if code compiles otherwise r = −10.

We train on two configurations for 20 epochs each:

- • RL 1: RL on the cadrille RL train set (selected parts from DeepCAD and Fusion360 train splits, absent from the SFT corpus).
- • RL 2: same as RL 1 + MCB train split. MCB is re-split by us so the test set covers all ISO categories and is never used in RL.

MCB exhibits a distinct rendering domain: mesh export uses relatively high STL tolerances that smooth sharp edges, producing softer silhouettes than DeepCAD and Fusion360. This alters view-space cues and introduces a domain shift. RL1 reproduces the cadrille protocol (RL on DeepCAD+Fusion360 only) for a like-for-like comparison; we expected weaker generalization to MCB under this shift. RL2 adds the MCB train split to the RL pool—while keep-

Table 2. Dataset/Regime definitions used in Table 1. “Canon.” = canonicalized (centered, scaled, quantized).

Tag Sup. Composition / Notes (view protocol)

cadrille SFT/RL SFT/RL Published baseline; RL with Dr.CPPO; 4 isometric views.

CADEvolve-P pre-aug

SFT Traced generator scripts only (CADEvolve-P); no code rewrites; no ABC/ShapeNet; canon. off; 7 views (6 ortho + 1 iso).

CADEvolve-P post-aug

SFT + Semantics-preserving code rewrites; pruned non-contributing ops; still no ABC/ShapeNet; canon. off; 7 views.

CADEvolve-C small

SFT Generator-only regime, canonicalized; 8 views (6 ortho + 2 iso).

CADEvolve-C middle

SFT + ABC/ShapeNet predictions (canonicalized); no CADRecode mix; 8 views.

CADEvolve-C big

SFT Middle + mix with CADRecode (canonicalized); 8 views.

CADEvolve-C big (RL1)

RL SFT init (C big) → RL on cadrille RL train (DeepCAD+Fusion360); 8 views.

CADEvolve-C big (RL2)

RL As RL1 + MCB train split; 8 views.

ing our MCB test fixed—to explicitly adapt to this smoother visual regime.

Baseline choice. We benchmark primarily against cadrille because it is the state-of-the-art in the image-only setting, offers public code, uses the same RL algorithm (Dr.CPPO) and matching evaluation metrics.

#### 4.5. Results

Table 1 summarizes Image2CAD performance on DEEPCAD, FUSION360, and MCB. We report Median CD↓ (×103), mean IoU↑ (%), and IR↓ (invalid rate).

We compare (i) the cadrille baselines (SFT/RL), (ii) CADEVOLVE-P before/after code-level augmentation, and (iii) CADEVOLVE-C under progressively stronger data

regimes (small/middle/big), followed by RL fine-tuning from the CADEVOLVE-C BIG SFT checkpoint (RL1/RL2).

Even after code-level augmentation, CADEVOLVE-P POST-AUG (SFT) remains far from the strongest baselines , although augmentation does move metrics in the right direction compared to pre-augmentation . This supports the interpretation that semantics-preserving rewrites reduce template overfitting, but are insufficient without canonization of the code style and shape size.

Moving from CADEVOLVE-C SMALL to MIDDLE yields a large jump , indicating that adding predictionderived supervision (ABC/ShapeNet) substantially improves geometric fidelity. The BIG regime further improves performance.

Starting from CADEVOLVE-C BIG (SFT), RL finetuning produces strong results across all three datasets. In RL1 improves over cadrille RL in CD/IoU on all datasets , at the cost of a slightly higher invalid rate, consistent with more frequent use of complex, collision-prone operations. RL2 augments the RL pool with MCB training shapes to address its domain shift (softer silhouettes due to higher STL tolerances). This yields a substantial improvement on MCB while maintaining near-cadrille performance on DeepCAD and Fusion360 .

### 5. Limitations

While CADEvolve is designed to provide a large-scale, diverse corpus of validated parametric CAD programs spanning a broad range of operators and geometric complexity, it is important to acknowledge several limitations of the proposed generation process and resulting dataset.

Synthetic distribution mismatch. CADEvolve is a synthetic dataset produced by an evolution loop and is not intended to match any single proprietary industrial CAD distribution. Consequently, the induced shape and operation frequencies may differ from real-world data. Despite this, we observe improved generalization across multiple benchmarks including stronger performance on MCB in our experiments, but we do not claim distribution-level fidelity to any particular industrial domain.

CadQuery dialect scope. The generated programs are expressed in CadQuery. Many operations are conceptually portable (e.g., extrude, revolve, loft, sweep, fillet/chamfer, booleans), but faithful conversion to other CAD systems may be non-trivial due to differences in feature-history representations, kernel behaviors, and constraint semantics.

### 6. Conclusion

We proposed CADEvolve, a general method for synthesizing high-quality supervision when open corpora are scarce, and instantiated it in the CAD domain. The resulting CADEvolve-3L dataset is the first CAD sequence

corpus covering the full CAD operation set, and it already yields state-of-the-art Image2CAD performance, suggesting that the same data foundation can further boost PC2CAD/Scan2CAD, Text2CAD, and broader multimodal CAD pipelines.

### References

- [1] Kamel Alrashedy, Pradyumna Tambwekar, Zulfiqar Haider Zaidi, Megan Langwasser, Wei Xu, and Matthew Gombolay. Generating cad code with vision-language models for 3d designs. In The Thirteenth International Conference on Learning Representations. 2
- [2] CadQuery Authors. Cadquery/cadquery: Cadquery 2.4.0,

2024. 1

- [3] Akshay Badagabettu, Sai Sravan Yarlagadda, and Amir Barati Farimani. Query2cad: Generating cad models using natural language queries. arXiv preprint arXiv:2406.00144, 2024. 2
- [4] Angel X Chang, Thomas Funkhouser, Leonidas Guibas, Pat Hanrahan, Qixing Huang, Zimo Li, Silvio Savarese, Manolis Savva, Shuran Song, Hao Su, et al. Shapenet: An information-rich 3d model repository. arXiv preprint arXiv:1512.03012, 2015. 2
- [5] Cheng Chen, Jiacheng Wei, Tianrun Chen, Chi Zhang, Xiaofeng Yang, Shangzhan Zhang, Bingchen Yang, ChuanSheng Foo, Guosheng Lin, Qixing Huang, et al. Cadcrafter: Generating computer-aided design models from unconstrained images. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025. 1, 3
- [6] Tianrun Chen, Chunan Yu, Yuanqi Hu, Jing Li, Tao Xu, Runlong Cao, Lanyun Zhu, Ying Zang, Yong Zhang, Zejian Li, et al. Img2cad: Conditioned 3d cad model generation from single image with structured visual geometry. arXiv preprint arXiv:2410.03417, 2024. 2
- [7] Taco S Cohen and Max Welling. Steerable cnns. arXiv preprint arXiv:1612.08498, 2016. 13
- [8] Anna C Doris, Md Ferdous Alam, Amin Heyrani Nobari, and Faez Ahmed. Cad-coder: An open-source vision-language model for computer-aided design code generation. arXiv preprint arXiv:2505.14646, 2025. 1, 2, 3
- [9] Tao Du, Jeevana Priya Inala, Yewen Pu, Andrew Spielberg, Adriana Schulz, Daniela Rus, Armando Solar-Lezama, and Wojciech Matusik. Inversecsg: Automatic conversion of 3d models to csg trees. ACM Transactions on Graphics (TOG), 37(6):1–16, 2018. 2
- [10] Kevin Ellis, Maxwell Nye, Yewen Pu, Felix Sosa, Josh Tenenbaum, and Armando Solar-Lezama. Write, execute, assess: Program synthesis with a repl. Advances in Neural Information Processing Systems, 32, 2019.
- [11] Markus Friedrich, Pierre-Alain Fayolle, Thomas Gabor, and Claudia Linnhoff-Popien. Optimizing evolutionary csg tree extraction. In Proceedings of the Genetic and Evolutionary Computation Conference, pages 1183–1191, 2019. 2
- [12] Yandong Guan, Xilin Wang, Xingxi Ming, Jing Zhang, Dong Xu, and Qian Yu. Cad-coder: Text-to-cad generation

- with chain-of-thought and geometric reward. arXiv preprint arXiv:2505.19713, 2025. 3
- [13] Haoxiang Guo, Shilin Liu, Hao Pan, Yang Liu, Xin Tong, and Baining Guo. Complexgen: Cad reconstruction by b-rep chain complex generation. ACM Transactions on Graphics (TOG), 41(4):1–18, 2022. 2
- [14] Nikolaus Hansen. The cma evolution strategy: A tutorial,

2023. 4

- [15] Changqi He, Shuhan Zhang, Liguo Zhang, and Jiajun Miao. Cad-coder: Text-guided cad files code generation. arXiv preprint arXiv:2505.08686, 2025. 2
- [16] Pradeep Kumar Jayaraman, Joseph G Lambourne, Nishkrit Desai, Karl DD Willis, Aditya Sanghi, and Nigel JW Morris. Solidgen: An autoregressive model for direct b-rep synthesis. Transactions on Machine Learning Research, 2023. 2
- [17] Kacper Kania, Maciej Zieba, and Tomasz Kajdanowicz. Ucsg-net-unsupervised discovering of constructive solid geometry tree. Advances in neural information processing systems, 33:8776–8786, 2020. 2
- [18] Mohammad Sadil Khan, Elona Dupont, Sk Aziz Ali, Kseniya Cherenkova, Anis Kacem, and Djamila Aouada. Cad-signet: Cad language inference from point clouds using layer-wise sketch instance guided attention. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4713–4722, 2024. 1, 2
- [19] Mohammad Sadil Khan, Sankalp Sinha, Talha Uddin, Didier Stricker, Sk Aziz Ali, and Muhammad Zeshan Afzal. Text2cad: Generating sequential cad designs from beginnerto-expert level text prompts. Advances in Neural Information Processing Systems, 37:7552–7579, 2024. 2
- [20] Sangpil Kim, Hyung-gun Chi, Xiao Hu, Qixing Huang, and Karthik Ramani. A large-scale annotated mechanical components benchmark for classification and retrieval tasks with deep neural networks. In Proceedings of 16th European Conference on Computer Vision (ECCV), 2020. 2
- [21] Sebastian Koch, Albert Matveev, Zhongshi Jiang, Francis Williams, Alexey Artemov, Evgeny Burnaev, Marc Alexa, Denis Zorin, and Daniele Panozzo. Abc: A big cad model dataset for geometric deep learning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 9601–9611, 2019. 2
- [22] Maksim Kolodiazhnyi, Denis Tarasov, Dmitrii Zhemchuzhnikov, Alexander Nikulin, Ilya Zisman, Anna Vorontsova, Anton Konushin, Vladislav Kurenkov, and Danila Rukhovich. cadrille: Multi-modal cad reconstruction with online reinforcement learning. arXiv preprint arXiv:2505.22914, 2025. 3, 7, 8
- [23] Joseph G Lambourne, Karl DD Willis, Pradeep Kumar Jayaraman, Aditya Sanghi, Peter Meltzer, and Hooman Shayani. Brepnet: A topological message passing system for solid models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 12773– 12782, 2021. 2
- [24] Joseph George Lambourne, Karl Willis, Pradeep Kumar Jayaraman, Longfei Zhang, Aditya Sanghi, and Kamal Rahimi Malekshan. Reconstructing editable prismatic cad from rounded voxel models. In SIGGRAPH Asia 2022 Conference Papers, pages 1–9, 2022. 2

- [25] Jiahao Li, Weijian Ma, Xueyang Li, Yunzhong Lou, Guichun Zhou, and Xiangdong Zhou. Cad-llama: leveraging large language models for computer-aided design parametric 3d model generation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 18563–18573,

2025. 2, 3

- [26] Lingxiao Li, Minhyuk Sung, Anastasia Dubrovina, Li Yi, and Leonidas J Guibas. Supervised fitting of geometric primitives to 3d point clouds. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 2652–2660, 2019. 2
- [27] Xueyang Li, Jiahao Li, Yu Song, Yunzhong Lou, and Xiangdong Zhou. Seek-cad: A self-refined generative modeling for 3d parametric cad using local inference via deepseek. arXiv preprint arXiv:2505.17702, 2025. 1, 2, 3
- [28] Yuan Li, Cheng Lin, Yuan Liu, Xiaoxiao Long, Chenxu Zhang, Ningna Wang, Xin Li, Wenping Wang, and Xiaohu Guo. Caddreamer: Cad object generation from single-view images. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 2025. 2
- [29] Zhihang Lin, Mingbao Lin, Yuan Xie, and Rongrong Ji. Cppo: Accelerating the training of group relative policy optimization-based reasoning models. arXiv preprint arXiv:2503.22342, 2025. 8
- [30] Yilin Liu, Jiale Chen, Shanshan Pan, Daniel Cohen-Or, Hao Zhang, and Hui Huang. Split-and-fit: Learning b-reps via structure-aware voronoi partitioning. ACM Transactions on Graphics (TOG), 43(4):1–13, 2024. 2
- [31] Yujia Liu, Anton Obukhov, Jan Dirk Wegner, and Konrad Schindler. Point2cad: Reverse engineering cad models from 3d point clouds. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 3763–3772, 2024. 2
- [32] Zichen Liu, Changyu Chen, Wenjun Li, Penghui Qi, Tianyu Pang, Chao Du, Wee Sun Lee, and Min Lin. Understanding r1-zero-like training: A critical perspective. arXiv preprint arXiv:2503.20783, 2025. 8
- [33] Weijian Ma, Shuaiqi Chen, Yunzhong Lou, Xueyang Li, and Xiangdong Zhou. Draw step by step: Reconstructing cad construction sequences from point clouds via multimodal diffusion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 27154– 27163, 2024. 2
- [34] Dimitrios Mallis, Ahmet Serdar Karadeniz, Sebastian Cavada, Danila Rukhovich, Niki Foteinopoulou, Kseniya Cherenkova, Anis Kacem, and Djamila Aouada. Cadassistant: Tool-augmented vllms as generic cad task solvers? arXiv preprint arXiv:2412.13810, 2024. 2
- [35] Chandrakana Nandi, James R Wilcox, Pavel Panchekha, Taylor Blau, Dan Grossman, and Zachary Tatlock. Functional programming for compiling and decompiling computer-aided design. Proceedings of the ACM on Programming Languages, 2(ICFP):1–31, 2018. 2
- [36] Alexander Novikov, Ngˆan Vu, Marvin Eisenberger, Emilien Dupont, Po-Sen Huang, Adam Zsolt Wagner, Sergey Shirobokov, Borislav Kozlovskii, Francisco J. R. Ruiz, Abbas Mehrabian, M. Pawan Kumar, Abigail See, Swarat Chaudhuri, George Holland, Alex Davies, Sebastian Nowozin,

- Pushmeet Kohli, and Matej Balog. AlphaEvolve: A coding agent for scientific and algorithmic discovery. arXiv, 2025. 2, 3
- [37] OpenAI. Gpt-5 system card. https://cdn.openai.com/gpt-5system-card.pdf, 2025. Accessed 2025-11-09. 2
- [38] Tobias Preintner, Weixuan Yuan, Adrian Konig, Thomas Back, Elena Raponi, and Niki van Stein. Evocad: Evolutionary cad code generation with vision language models. arXiv,

2025. 2, 3

- [39] Daxuan Ren, Jianmin Zheng, Jianfei Cai, Jiatong Li, Haiyong Jiang, Zhongang Cai, Junzhe Zhang, Liang Pan, Mingyuan Zhang, Haiyu Zhao, et al. Csg-stump: A learning friendly csg-like representation for interpretable shape parsing. In Proceedings of the IEEE/CVF international conference on computer vision, pages 12478–12487, 2021. 2
- [40] Daxuan Ren, Jianmin Zheng, Jianfei Cai, Jiatong Li, and Junzhe Zhang. Extrudenet: Unsupervised inverse sketchand-extrude for shape parsing. In European Conference on Computer Vision, pages 482–498. Springer, 2022. 2
- [41] Bernardino Romera-Paredes. Mathematical discoveries from program search with large language models. Nature 625.7995, page 468–475, 2024. 3
- [42] Danila Rukhovich, Elona Dupont, Dimitrios Mallis, Kseniya Cherenkova, Anis Kacem, and Djamila Aouada. Cad-recode: Reverse engineering cad code from point clouds. arXiv preprint arXiv:2412.14042, 2024. 1, 2
- [43] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Y Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024. 8
- [44] Gopal Sharma, Rishabh Goyal, Difan Liu, Evangelos Kalogerakis, and Subhransu Maji. Csgnet: Neural shape parser for constructive solid geometry. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 5515–5523, 2018. 3
- [45] Gopal Sharma, Difan Liu, Subhransu Maji, Evangelos Kalogerakis, Siddhartha Chaudhuri, and Radom´ır Mˇech. Parsenet: A parametric surface fitting network for 3d point clouds. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part VII 16, pages 261–276. Springer, 2020. 2
- [46] Patrice Y Simard, David Steinkraus, John C Platt, et al. Best practices for convolutional neural networks applied to visual document analysis. In Icdar. Edinburgh, 2003. 13
- [47] Yonglong Tian, Andrew Luo, Xingyuan Sun, Kevin Ellis, William T. Freeman, Joshua B. Tenenbaum, and Jiajun Wu. Learning to infer and execute 3d shape programs. In International Conference on Learning Representations, 2019. 2
- [48] Niki van Stein and Thomas B¨ack. Llamea: A large language model evolutionary algorithm for automatically generating metaheuristics. arXiv, 2024. 3
- [49] Niki van Stein, Diederick Vermetten, and Thomas B¨ack. Inthe-loop hyper-parameter optimization for llm-based automated design of heuristics. ACM Transactions on Evolutionary Learning and Optimization, 2024. 3
- [50] Jianing Wang, Junda Wu, Yupeng Hou, Yao Liu, Ming Gao, and Julian McAuley. Instructgraph: Boosting large language

- models via graph-centric instruction tuning and preference alignment. arXiv, 2024. 3
- [51] Kehan Wang, Jia Zheng, and Zihan Zhou. Neural face identification in a 2d wireframe projection of a manifold object. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1622–1631, 2022. 2
- [52] Ruiyu Wang, Yu Yuan, Shizhao Sun, and Jiang Bian. Textto-cad generation through infusing visual feedback in large language models. arXiv preprint arXiv:2501.19054, 2025. 3
- [53] Siyu Wang, Cailian Chen, Xinyi Le, Qimin Xu, Lei Xu, Yanzhou Zhang, and Jie Yang. Cad-gpt: Synthesising cad construction sequence with spatial reasoning-enhanced multimodal llms. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 7880–7888, 2025. 2
- [54] Xiaogang Wang, Yuelang Xu, Kai Xu, Andrea Tagliasacchi, Bin Zhou, Ali Mahdavi-Amiri, and Hao Zhang. Pie-net: Parametric inference of point cloud edges. Advances in neural information processing systems, 33:20167–20178, 2020. 2
- [55] Yinghui Wang, Xinyu Zhang, and Peng Du. Gacocad: Geometry-augmented and conciseness-optimized cad model generation from single image. arXiv preprint arXiv:2510.17157, 2025. 1, 3
- [56] Maurice Weiler, Mario Geiger, Max Welling, Wouter Boomsma, and Taco S Cohen. 3d steerable cnns: Learning rotationally equivariant features in volumetric data. Advances in Neural information processing systems, 31, 2018. 13
- [57] Karl DD Willis, Yewen Pu, Jieliang Luo, Hang Chu, Tao Du, Joseph G Lambourne, Armando Solar-Lezama, and Wojciech Matusik. Fusion 360 gallery: A dataset and environment for programmatic cad construction from human design sequences. ACM Transactions on Graphics (TOG), 40(4): 1–24, 2021. 1, 2
- [58] Rundi Wu, Chang Xiao, and Changxi Zheng. Deepcad: A deep generative network for computer-aided design models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 6772–6782, 2021. 1, 2
- [59] Xiang Xu, Karl DD Willis, Joseph G Lambourne, Chin-Yi Cheng, Pradeep Kumar Jayaraman, and Yasutaka Furukawa. Skexgen: Autoregressive generation of cad construction sequences with disentangled codebooks. In International Conference on Machine Learning, pages 24698–24724. PMLR, 2022.
- [60] Xiang Xu, Pradeep Kumar Jayaraman, Joseph G Lambourne, Karl DD Willis, and Yasutaka Furukawa. Hierarchical neural coding for controllable cad model generation. In International Conference on Machine Learning, pages 38443– 38461, 2023. 2
- [61] Xiang Xu, Joseph Lambourne, Pradeep Jayaraman, Zhengqing Wang, Karl Willis, and Yasutaka Furukawa. Brepgen: A b-rep generative diffusion model with structured latent geometry. ACM Transactions on Graphics (TOG), 43

(4):1–14, 2024. 2

- [62] Xiaolong Yin, Xingyu Lu, Jiahang Shen, Jingzhe Ni, Hailong Li, Ruofeng Tong, Min Tang, and Peng Du. Rlcad: Reinforcement learning training gym for revolution in-

- volved cad command sequence generation. arXiv preprint arXiv:2503.18549, 2025. 1, 3
- [63] Fenggen Yu, Zhiqin Chen, Manyi Li, Aditya Sanghi, Hooman Shayani, Ali Mahdavi-Amiri, and Hao Zhang. Capri-net: Learning compact cad shapes with adaptive primitive assembly. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 11768– 11778, 2022. 2
- [64] Fenggen Yu, Qimin Chen, Maham Tanveer, Ali Mahdavi Amiri, and Hao Zhang. D2csg: Unsupervised learning of compact csg trees with dual complements and dropouts. Advances in Neural Information Processing Systems, 36: 22807–22819, 2023. 2
- [65] Yu Yuan, Shizhao Sun, Qi Liu, and Jiang Bian. Cadeditor: A locate-then-infill framework with automated training data synthesis for text-based cad editing. arXiv preprint arXiv:2502.03997, 2025. 2
- [66] Zeqing Yuan, Haoxuan Lan, Qiang Zou, and Junbo Zhao. 3d-premise: Can large language models generate 3d shapes with sharp features and parametric control? arXiv preprint arXiv:2401.06437, 2024. 2
- [67] Zhe Yuan, Jianqi Shi, and Yanhong Huang. Openecad: An efficient visual language model for editable 3d-cad design. Computers & Graphics, 124:104048, 2024. 3
- [68] Chao Zhang, Arnaud Polette, Romain PINQUIE,´ Mirai Iida, Henri De Charnace, and Jean-Philippe Pernot. Reinforcement learning-based parametric cad models reconstruction from 2d orthographic drawings. Available at SSRN 5174280,

2025. 3

- [69] Zhanwei Zhang, Shizhao Sun, Wenxiao Wang, Deng Cai, and Jiang Bian. Flexcad: Unified and versatile controllable cad generation with fine-tuned large language models. International Conference on Learning Representations, 2025. 2
- [70] Dmitrii Zhemchuzhnikov. Volumetric Analysis and Arbitrary-Shaped Pattern Recognition in Neural Networks Using Fourier Domain Representations. PhD thesis, Universit´e Grenoble Alpes [2020-....], 2024. 13
- [71] Dmitrii Zhemchuzhnikov and Sergei Grudinin. On the fourier analysis in the so (3) space: the equilopo network. In The Thirteenth International Conference on Learning Representations.
- [72] Dmitrii Zhemchuzhnikov and Sergei Grudinin. Ilpo-net: Network for the invariant recognition of arbitrary volumetric patterns in 3d. In Joint European Conference on Machine Learning and Knowledge Discovery in Databases, pages 352–368. Springer, 2024.
- [73] Dmitrii Zhemchuzhnikov, Ilia Igashov, and Sergei Grudinin. 6dcnn with roto-translational convolution filters for volumetric data processing. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 4707–4715, 2022. 13

### A. Evolutionary synthesis trajectories

Fig. 7 visualizes representative trajectories produced by our evolutionary propose–execute–filter pipeline driven by a LLM. Starting from a small set of simple seed primitives (top), the model proposes incremental code edits that introduce new operations and structural detail. Accepted candidates become parents for subsequent iterations, yielding branching lineages and occasional recombination across different design directions. Overall, the graph illustrates how complexity is accumulated progressively — from basic solids to multi-operation, higher-detail parts — under automated execution and validation constraints.

[Figure 221]

[Figure 222]

[Figure 223]

[Figure 224]

[Figure 225]

[Figure 226]

[Figure 227]

[Figure 228]

[Figure 229]

Figure 7. Example evolutionary trajectories. Nodes correspond to generated CAD parts; directed edges indicate parent→child refinements proposed by the LLM. Over iterations, the shapes become progressively more complex through compositional multioperation edits.

### B. Novelty–validity dynamics

Understanding the evolution of novelty and validity metrics during search is essential for analyzing both dataset quality and sampling efficiency. In our runs, the search process did not terminate due to full saturation of the design space, but rather due to a practical constraint: under strict validation rules, the rate of invalid proposals increases rapidly in later iterations, reaching up to ∼85%. At the same time, the acceptance rate of novel samples drops to 40– 50%, indicating diminishing returns under a fixed API or compute budget. These trends are visualized in Fig. 8 and Fig. 9. The former shows the steadily increasing invalidity rate as the search progresses, while the latter highlights the decreasing share of accepted novel samples. Together, they illustrate a key trade-off: although exploration can continue, its efficiency degrades substantially. Extending the process further would likely require stronger proposal strategies rather than simply running longer.

### C. Rotational augmentation

We apply rotational augmentation to make training robust to global orientation. In practice, the same CAD part can be stored or observed under arbitrary rotations, while its construction logic and parameterization remain unchanged. Without augmentation, the

1.0

0.8

Invalidratio

0.6

0.4

0.2

0.0

0 200 400 600 800

step

###### Figure 8. Invalid proposal rate over search iterations.

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

1.0

0.9

0.8

Noveltyratio

0.7

0.6

0.5

0.4

0.3

0.2

0 200 400 600 800

step

###### Figure 9. Novelty acceptance rate over search iterations.

model may implicitly rely on dataset-specific canonical poses, which reduces generalization [7, 46, 56, 70–73]. By adding randomly rotated variants of each script, we encourage the model to focus on pose-independent geometric and procedural cues and improve performance on unseen orientations. To perform rotational augmentation, we use a script-based rotation procedure. Specifically, we rotate the arguments of CADQUERY Workplane construction calls, which changes the orientation and offsets of the reference planes used by subsequent operations. All remaining calls are divided into two categories: local-coordinate and globalcoordinate. Local-coordinate functions operate in the coordinate frame of a given workplane and therefore require no changes under rotation. Global-coordinate functions, in contrast, are defined in the global frame and must be rotated accordingly. The algorithm thus detects workplane-creation and global-coordinate calls and rewrites their arguments by applying the corresponding rotation. We consider 24 rotation variants, grouped into three types:

- 1. rotation by 0◦, 90◦, 180◦, or 270◦ about the Z-axis;
- 2. rotation by 0◦, 90◦, 180◦, or 270◦ about the Z-axis, followed by a 90◦ rotation about the Y -axis, and then by 0◦, 90◦, 180◦, or 270◦ about the Z-axis;
- 3. rotation by 0◦, 90◦, 180◦, or 270◦ about the Z-axis, followed by a 180◦ rotation about the Y -axis.

For each dataset element, one random rotation was applied and the resulting sample was added to the training set.

### D. Comparison to existing benchmarks

To characterize CADEvolve and compare it to existing CAD benchmarks, we report three complementary statistics: (i) operation occurrence (Table 3), (ii) sequence length—the number of

CAD operations per script (Fig. 10), and (iii) face count—the number of polygonal faces in the resulting geometry (Fig. 11). Together, these metrics capture operator coverage, procedural depth, and geometric complexity.

Operation occurrence statistics. Table 3 reports the fraction of scripts that contain each CADQUERY operation. Overall, the distribution broadly follows that of real CAD program histories, with two notable shifts: (i) fewer revolve, chamfer, shell, and mirror operations; and (ii) more hole operations and substantially more transform and loft operations. Despite these differences, the most frequent operators are present in sufficient quantities for reliable training and evaluation.

Table 3. Operation statistics.

Operation % extrude 83.05% fillet 27.78% revolve 4.80% chamfer 4.76% hole 11.99% shell 1.95% mirror 0.08% sweep 5.75% transform 20.45% loft 8.48%

Sequence length. As shown in Fig. 10, CADEvolve exhibits a wide distribution of program lengths with a long tail of highly procedural models, indicating substantially greater procedural depth than typical benchmarks.

40

35

30

4Count(×10)

25

20

15

10

5

0

1 18 35 52 69 86 103 120 137

Sequence Length

Figure 10. Sequence length distribution. Many CADEvolve scripts exceed 25 operations, with a long tail of highly procedural models.

Face count. Fig. 11 shows that CADEvolve parts frequently contain thousands of polygonal faces, reflecting fine-grained geometric detail and higher shape complexity than existing benchmarks.

0.10

| | |
|---|---|
| | |
| | |

0.08

0.06

Density

0.04

0.02

0

100 101 102 103 104 105 106 107 Faces (log scale)

Figure 11. Face count distribution. CADEvolve parts frequently contain thousands of faces, reflecting fine-grained and detailed geometry.

