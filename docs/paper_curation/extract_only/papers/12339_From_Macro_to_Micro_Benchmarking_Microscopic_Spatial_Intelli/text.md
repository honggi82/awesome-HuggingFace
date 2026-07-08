## arXiv:2512.10867v2[cs.CV]12Dec2025

### From Macro to Micro: Benchmarking Microscopic Spatial Intelligence on Molecules via Vision-Language Models

Zongzhao Li1,2,3* Xiangzhe Kong4,5* Jiahui Su6 Zongyang Ma7 Mingze Li1,2,3 Songyou Li1,2,3 Yuelin Zhang1,2,3 Yu Rong8,9 Tingyang Xu8,9 Deli Zhao8,9 Wenbing Huang1,2,3† 1Gaoling School of Artificial Intelligence, Renmin University of China 2Beijing Key Laboratory of Research on Large Models and Intelligent Governance 3Engineering Research Center of Next-Generation Intelligent Search and Recommendation, MOE 4Dept. of Comp. Sci. & Tech., Tsinghua University 5Institute for AI Industry Research (AIR), Tsinghua University 6SKL-ESPC & SEPKL-AERM, College of Environmental Sciences and Engineering, Peking University 7MAIS, Institute of Automation, Chinese Academy of Sciences 8DAMO Academy, Alibaba Group, Hangzhou, China 9Hupan Lab, Hangzhou, China

lizongzhao2023@ruc.edu.cn, jackie kxz@outlook.com, hwenbing@126.com

##### Abstract

This paper introduces the concept of Microscopic Spatial Intelligence (MiSI), the capability to perceive and reason about the spatial relationships of invisible microscopic entities, which is fundamental to scientific discovery. To assess the potential of Vision-Language Models (VLMs) in this domain, we propose a systematic benchmark framework MiSI-Bench. This framework features over 163,000 question-answer pairs and 587,000 images derived from approximately 4,000 molecular structures, covering nine complementary tasks that evaluate abilities ranging from elementary spatial transformations to complex relational identifications. Experimental results reveal that current state-of-the-art VLMs perform significantly below human level on this benchmark. However, a fine-tuned 7B model demonstrates substantial potential, even surpassing humans in spatial transformation tasks, while its poor performance in scientifically-grounded tasks like hydrogen bond recognition underscores the necessity of integrating explicit domain knowledge for progress toward scientific AGI. The datasets are available at https://huggingface.co/ datasets/zongzhao/MiSI-bench.

##### 1. Introduction

Spatial intelligence [11, 46], a critical component of advanced artificial intelligence, empowers systems to per-

*Equal contribution. †Corresponding author.

ceive, interpret, and interact with the three-dimensional world. Such capability necessitates a profound comprehension of geometries, spacial relationships, and even fundamental physical rules. Contemporary efforts address this by utilizing Vision-Language Models (VLMs) [6, 27, 28], which jointly process visual and textual data to learn spatial properties and relationships from complex scenes [18, 45, 47, 49]. This reasoning capacity regarding object layouts, occlusions, and perspectives establishes a vital foundation for embodied interaction in real-world environments [50].

Yet, beyond this visible macroscopic realm lies the microscopic world, composed of invisible particles (e.g., atoms and molecules) that constitute matter [16], where spatial reasoning takes on a profoundly different form. In molecular sciences, experts routinely visualize and manipulate microscopic entities such as proteins and drugs using software tools (e.g., PyMOL [15], ChimeraX [34]) to explore geometric complementarity, analyze interactions, and design new functional molecules. This process relies on a specialized form of spatial reasoning: the ability to reconstruct three-dimensional structures from two-dimensional projections and to infer physical relationships (e.g., hydrogen bonds). In this paper, we refer to this capability as Microscopic Spatial Intelligence (MiSI), the cognitive foundation underlying human expertise and discovery in scientific fields such as structural biology, drug discovery, and material design.

The success of Large Language Models (LLMs) in general-purpose tasks has spurred their exploration in scientific discovery [8, 30, 39], motivating the use of VLMs

[Figure 1]

[Figure 2]

###### STEP 1: Unit Tasks STEP 2: Composite Tasks

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

[Figure 17]

Rotation + Rotation

Zooming

[Figure 18]

[Figure 19]

[Figure 20]

|[Figure 21]<br><br>[Figure 22]<br><br>[Figure 23]|
|---|

|[Figure 24]<br><br>[Figure 25]<br><br>[Figure 26]|
|---|

Translation, Rotation, Residue-ligand Interaction

[Figure 27]

[Figure 28]

[Figure 29]

Docking

[Figure 30]

[Figure 31]

[Figure 32]

|[Figure 33]|
|---|

|[Figure 34]<br><br>[Figure 35]<br><br>[Figure 36]|
|---|

- 1ebw 2gvj

1c86 1dzk

- 2bsu 4yc0 …………

Binding Pocket

[Figure 37]

[Figure 38]

[Figure 39]

Interaction Location

[Figure 40]

[Figure 41]

[Figure 42]

|[Figure 43]|
|---|

|[Figure 44]|
|---|

Pocket-ligand Interaction

#### 4000+ PDBs 9 Microscopic Spatial tasks

#### 588K data

Figure 1. Overview of our MiSI-Bench. Our dataset is derived from around 4,000 PDBs and comprises 9 distinct tasks.

to analyze scientific data. VLMs are uniquely suited for this role, as they can process both visual and textual modalities within a unified architecture. Unlike conventional domain-specific systems, VLMs can perceive structural patterns while grounding their interpretations in scientific concepts. This cross-modal capability enables a more humanlike, context-aware reasoning about molecular structures by seamlessly linking them with textual semantics. However, shifting from human-scale daily objects to atom-level invisible entities, MiSI requires exceptional scientific expertise to perceive spatial transformations and reason over relational identifications such as atomic interactions. It remains unclear whether the VLMs are ready for tackling the challenges in the microscopic scientific fields.

To bridge the gap, we propose MiSI-Bench, a systematical framework for training and evaluating microscopic spatial intelligence in VLMs. As illustrated in Fig. 1, MiSI-Bench contains 163,514 question-answer pairs and 587,975 images over a diverse set of 9 complementary tasks, constructed from around 4,000 molecular structures [44]. We visualize these three-dimensional microscopic objects as two-dimensional orthographic projections, just like how human experts interpret them. We then disentangle the intelligence for spatial transformations and relational identifications into four elementary operations: trans-

lation, rotation, zooming, interaction. Subsequently, we design four unit tasks to evaluate these fundamental abilities independently, and further design five composite tasks which integrate multiple elementary operations to test the models’ high-order reasoning ability.

Experimental results show that current advanced VLMs (e.g., o3 [33], Claude Sonnet4.5 [5]) perform well below human level on our benchmark. While human evaluators excel in some tasks, they struggle with continuous spatial transformation and 3D reconstruction. Remarkably, after SFT on our dataset, a 7B model outperforms all leading VLMs and even surpasses humans in spatial transformation tasks, revealing VLMs’ untapped potential for spatial reasoning. However, its poor performance in biologicallygrounded tasks like hydrogen bond recognition highlights the need for injecting explicit scientific knowledge during pre-training to progress toward AGI.

##### 2. Related Work

Macroscopic Spatial Intelligence In recent years, researchers have developed a variety of datasets and benchmarks with distinct focuses to evaluate the spatial intelligence of VLMs [20, 37, 51]. For instance, VIS-Bench [46] and MuriBench [43] emphasize the model’s ability to associate and reason across video/multi-images; LEGO-

Puzzles [40] examines multi-step spatial reasoning in a synthetic block-building environment. Therefore, we propose MiSI-Bench to draw attention to this direction and to establish a reliable benchmark for evaluating models’ microlevel spatial intelligence. This task is uniquely challenging, as understanding microscopic entities demands exceptional expertise in both spatial transformations and relational reasoning.

Three-Dimensional Molecular Understanding Conventional modeling of 3D molecules usually relies on Cartesian coordinates as model inputs. Starting from physical force fields [1, 38], models have evolved from 3D convolutional neural networks [35] to equivariant graph neural networks [23, 41] and transformers [22, 26] with the rise of geometric deep learning [9]. However, these approaches primarily operate within geometric coordinate space only, while MLLMs offer a complementary, human-like perspective which can learn to reason about three-dimensional molecular geometry through visual abstractions and natural language grounding, unlocking a new mode of molecular understanding that bridges microscopic visual perception and textual knowledge.

##### 3. Definition of Major Concepts

To explore whether current VLMs possess the ability to understand 3D molecular structures, which we term MiSI as above, we begin by defining the representations adopted for molecular structures, and then introduce the fundamental perceptual expertise humans rely on to comprehend the micro world. These elements together serve as the preliminaries for our benchmark design.

Study Scope Our physical world is organized across multiple hierarchical levels, from macroscopic organisms to organs, cells, and to molecules such as proteins, and DNAs [16]. Thanks to the continuous scientific exploration, people now understand the phenomena observed in the macroscopic world are the results of microscopic particles. And advances in imaging technologies, such as cryoelectron microscopy, further allow us to visualize these particles at near-atomic resolution [7]. In this work, we shift the focus from the macroscopic world to its microscopic foundation, investigating how VLMs perceive and reason about 3D molecular structures composed of atoms.

Orthographic Projection of Molecules Throughout human history, people have sought to represent the threedimensional world on two-dimensional media. One classical approach employs perpendicular rays of light to generate orthographic projections, and typically canonical views,

such as the front, top, and left views, are employed to reconstruct the full 3D structure of an object [10]. Following this convention, we adopt orthographic views as 2D representations of microscopic 3D molecular structures.

Taxonomy of Human Expertise Understanding microscopi molecules requires both spatial reasoning and domain expertise. Experts rely on fundamental spatial transformation abilities, such as translation, rotation, and zooming, to establish a more complete panorama of molecular structures [12, 25]. Beyond geometric manipulation, they use domain knowledge to identify interaction patterns such as hydrogen bonds, which reveal the underlying physical principles of molecular organization [2]. We refer to this process as relational identification. In this work, we summarize the expert skills with the above-mentioned four elementary microspace operations, namely translation, rotation, zooming, and interaction, then design unit tasks to independently evaluate each capability. We further introduce combinatorial tasks that require integrating multiple operations, enabling a more comprehensive assessment of VLMs in microspace understanding.

##### 4. MiSI-Bench

We construct our MiSI-Bench for evaluating VLMs using the refined PDBbind dataset [44], a widely adopted benchmark for structure-based drug discovery [21, 24, 42]. Each entity in PDBbind dataset corresponds to a complex composed of a protein and a ligand. We visualize all complexes in ChimeraX [34] to generate orthographic projections images as model inputs. After removing complexes with visualization issues, the final dataset contains 3,503 protein–ligand complexes for Supervised Fine-Tuning (SFT) and 490 for testing, all with experimentally solved crystal structures. The benchmark encompasses nine tasks, including four unit tasks involving single elementary operations and five composite tasks involving combinations of multiple operations. For each task, the QA pairs are generated using fixed templates. The problem templates and a brief illustration of all tasks are shown in Fig. 2. The overall pipeline for constructing MiSI-Bench is detailed in supplementary materials. Our benchmark contains a total of 150,597 Question Answering (QA) pairs for training and 12,917 for testing, summing up to 538,015 and 49,960 images in the train and the test set, respectively. The statistics for all tasks are presented in Fig. 3. Two formats of questions are used to evaluate model performance.

Cloze Questions require the model to complete partially specified instructions by filling in missing actions or parameters. These tasks assess the ability of the models to identify the correct operations with precise attributes (e.g., axes, distances, or angles).

[Figure 45]

[Figure 46]

[Figure 47]

- T1：Translation

[Figure 48]

[Figure 49]

- T2：Rotation

[Figure 50]

- T3：Zooming

T9：Pocket-ligand Interaction

T8：Interaction Location

[Figure 51]

[Figure 52]

|[Figure 53]<br><br>I will provide six images showing a protein-ligand complex (PDB: {PDB_ID}) from different views:<br><br>- Images 1–6: Front, left, top, back, right, and bottom perspectives ## Task:<br>- Locate the hydrogen bond components (chain {CHAIN}, residue {RESIDUE} {RES_NUM} and ligand atom {LIGAND_ATOM}).<br>- Calculate the centroid position of all atoms involved in this bond.<br>- Determine the translation required to move this centroid to the<br><br><br>screen center (0,0), then select the correct translation command from the provided options.<br><br>Translation to (0, 0, 0)<br><br>Hydrogen bond<br><br>Example: “ASN 460 ND2 A, O2”<br><br>Ligand + Protein<br><br>ASN 460|
|---|

I will provide you with four images showing the binding interaction between a protein (PDB ID: {PDB_ID}) and a small molecule ligand. ## Image Description:

I will provide you with six images showing the binding interaction between a specific protein residue (PDB ID: {PDB_ID}) and a small molecule ligand.

- - **Image 1**: Front view (main perspective)
- - **Image 2**: Left view (viewed from the left side)
- - **Image 3**: Top view (viewed from above)
- - **Image 4**: Front view after spatial transformation (the pocket-

- Images 1–6: Front, left, top, back, right, and bottom perspectives

## Task: Analyze the six images to identify all possible hydrogen bonds between

the pocket and the ligand

ligand complex has been transformed in 3D space and then observed from the front view perspective)

## Output Format: List all hydrogen bonds in the following format:

## Task: Please analyze the transformation from Image 1 to Image 4 and determine the required translation command to achieve this transformation.

**[Amino Acid Type] [Residue Number] [Donor/Acceptor Atom] [Chain ID], [Ligand Atom Name]**

##Output:

ASN 460

-the hydrogen bonds between ligand and protein are

ASN 460 ND2 A, O2; GLU 537 OE1 A, O2.

GLU537

Ligand + Protein

[Figure 54]

T7：Docking

[Figure 55]

I will provide seven images showing a protein binding pocket and a displaced ligand (PDB: {PDB_ID}). ## Image Description:

I will provide you with four images showing the binding interaction between a protein (PDB ID: {PDB_ID}) and a small molecule ligand. ## Image Description: Similar with T1.

- - Images 1–3: Binding pocket only (front, left, and top views)
- - Images 4–6: Ligand only in its displaced position (front, left, and

# Tasks

## Task: Please analyze the transformation from Image 1 to Image 4 and determine the required rotation command to achieve this transformation.

top views)

- Image 7: Combined front view showing both pocket and displaced ligand ## Task:

- - The ligand has been moved and rotated from its original pose.
- - Analyze the spatial relationship between the pocket (Images 1–3)

[Figure 56]

and the displaced ligand (Images 4–7).

I will provide you with four images showing the binding interaction between a protein (PDB ID: {PDB_ID}) and a small molecule ligand. ## Image Description: Similar with T1.

- Determine the sequence of operations required to dock the ligand back into the binding pocket.

## Task: Please analyze the transformation from Image 1 to Image 4 and determine the required scaling command to achieve this transformation.

Ligand

Dock

Protein

[Figure 57]

[Figure 58]

[Figure 59]

T4：Residue-ligand Interaction

T5：Translation + Rotation

T6：Rotation + Rotation

[Figure 60]

[Figure 61]

[Figure 62]

I will provide two sets of images depicting a protein–ligand complex.

I will provide two sets of images showing protein-ligand complexes.

I will provide you with six images showing the binding interaction between a specific protein residue (PDB ID: {PDB_ID}) and a small molecule ligand. ## Image Description:

- ## Part 1: Reference Transformation (4 images) Reference protein (PDB: {PDB_ID}) demonstrating a spatial transformation:

- - Images 1–3: Front, left, and top initial views
- - Image 4: Front view after two sequential rotations

- ## Part 2: Apply Transformation (4 images) The same protein for applying the transformation:

- ## Part 1: Reference Transformation (4 images) Reference protein (PDB: {REF_PDB_ID}) demonstrating a spatial transformation:

- - Images 1-3: Front, left, and top initial views
- - Image 4: Front view after transformation (1 rotation + 1 translation)

- ## Part 2: Apply Transformation (7 images) Target protein (PDB: {TARGET_PDB_ID}) for transformation application:

- - Image 1: Front view (main perspective)
- - Image 2: Left view (viewed from the left side)
- - Image 3: Top view (viewed from above)
- - Image 4: Back view (viewed from behind)
- - Image 5: Right view (viewed from the right side)
- - Image 6: Bottom view (viewed from below)

- Images 5–8: Four options (A, B, C, D) showing possible results ## Task:

- - Images 5-7: Front, left, and top initial views
- - Images 8-11: Four options (A, B, C, D) showing possible results

## Task:

Analyze the six images to determine if there are any hydrogen bonds between the target protein residue and the ligand. If hydrogen bonds exist, identify all of them.

- - Analyze the transformation from Images 1–3 to Image 4
- - Apply the same two rotations in the same order to Image 4
- - Select the correct result from options A–D

Hydrogen bond

## Task:

Residue

- - Analyze the transformation from Images 1-3 to Image 4
- - Apply the same transformation to the target protein
- - Select the correct result from options A-D

Ligand

## Output Format: Your answer should follow one of these three formats:

- - "No" (if no hydrogen bonds)
- - "Yes: VAL O, N1" (if one hydrogen bond)
- - "Yes: VAL O, N1; VAL O, N2" (if multiple hydrogen bonds)

List all hydrogen bonds in the following format:

Rotation Rotation

Translation Rotation

**[Amino Acid Type] [Donor/Acceptor Atom], [Ligand Atom Name]**

Figure 2. A brief demonstration of the MiSI-Bench dataset. The examples have been simplified for clarity; for complete examples, please refer to Appendix C.

[Figure 63]

Unit Tasks (48.29%)

Translation (9.30%) Rotation (13.95%) Zooming (9.30%) Residue-ligand Interaction (15.73%)

UnitTasks

Composite Tasks

MiSI

[Figure 64]

Composite Tasks (51.71%）

Translation + Rotation (13.95%) Rotation + Rotation (13.95%) Docking (13.95%) Interaction Location (7.52%) Pockte-ligand Interaction (2.33%)

Figure 3. The statistics for all tasks.

Multiple-Choice Questions present several candidate options, among which the model should identify the correct answer while rejecting distracting decoys. These tasks evaluate whether the models have discriminative understanding of spatial configurations and their ability to reason about the consequences of microspace operations.

###### 4.1. Unit Task

We first establish four unit tasks to evaluate the spatial understanding of elementary microspace operations, where each task isolates one essential ability involved in manipulating or interpreting microscopic 3D molecular structures. For translation, rotation, and zooming tasks, the three orthographic projections (i.e., the top, the front, and the left side

views) of the initial complex and the front view of the complex after the operation are given to the models. For residueligand interaction task, since overlapping atom names might interfere with the performance, we give all six orthographic projections to the models.

Translation (Cloze) In this task, the molecular complex is translated along one of the axes parallel to the visualization plane (i.e., the x or y axis) by a random distance between −4 and 4 angstrom (A).˚ The model must infer both the direction and the magnitude of motion, completing a prompt of the form: move x 3 . To avoid being too harsh on numerical precisions, the translation range is discretized into 1.0 A˚ bins. For SFT, two samples per complex are generated along each axis, yielding 14,012 training samples, and one per axis for evaluation, totaling 980 test samples.

Rotation (Cloze) The complex is rotated along one of the three coordinate axes (x, y, or z) by a random angle uniformly drawn from [−90◦,90◦]. Models must determine both the rotation axis and the degree of rotation, filling in the prompt: roll x 15 . Rotation angles are discretized into 15◦ bins. Each complex results in two samples per axis for SFT (21,018 training samples) and one per axis for testing (1,470 samples).

Zooming (Cloze) To simulate zooming operations, the complex is moved along the axis perpendicular to the visualization plane (i.e., the z axis) by a random depth between 40 and 60 A.˚ This range corresponds to the magnification levels most suitable for visualizing the pocket–ligand interactions near the center of the view (See the distribution figure in Appendix A for details). The model fills in prompts like: move z 50 , where depth values are discretized into 1.0 A˚ bins. Four samples per complex are created for SFT (14,012 training samples) and two per complex for testing (980 samples).

Residue-Ligand Interaction (Cloze) Given a residue and the ligand, models must first identify whether the residue interacts with the ligand (Yes or No), and then output all atom pairs participating in the interaction: ARG NH2, O22; ARG N, O23. For this proof-ofconcept benchmark, we focus on hydrogen bonds as interactions of interest, with detailed geometric configurations provided in the Appendix A. The dataset includes 11,572 positive and 12,125 negative samples for SFT, and 1,499 positive and 1,603 negative samples for evaluation.

###### 4.2. Composite Tasks

We further design five composite tasks that require models to understand and reason on multiple microspace oper-

ations, the capability of which are commonly required for human experts during molecular structural analysis.

###### 4.2.1. Spatial Transformation Reasoning

This category evaluates the ability of the models to reason about sequential spatial transformations and generalize them across different molecular complexes. Denote two complexes as c1 and c2, and two spatial transformations as

- f1 and f2. The models are given the three orthographic projections of both c1 and c2, along with the front view of f2(f1(c1)), which is the result of applying f1 followed by
- f2 to c1. The task is to identify the correct front view of f2(f1(c2)) among four candidate images, where the other three are decoys. The decoys are constructed by exerting perturbation on the ground-truth transformations through one of three schemes: 1) Altering the magnitude (translation distance or rotation angle) of both f1 and f2; 2) Flipping the sign of f1 (e.g., clockwise to counterclockwise) and adjusting the magnitude of f2; 3) Changing the axis of f1 and modifying the magnitude of f2.

Translation-Rotation Movement (Multiple-Choice) In this task, f1 is sampled from translational operations and f2 from rotational operations. For each complex in the dataset, we pair it with another random complex and construct six and three distinct transformation combinations for SFT and testing, respectively, yielding a total of 21,018 and 1,470 questions for SFT and testing.

Rotation-Rotation Movement (Multiple-Choice) Both f1 and f2 are sampled from rotational operations, constrained to different axes to prevent trivial correlations. The scale matches the previous task, with 21,018 questions for SFT and 1,470 for testing.

###### 4.2.2. Local Relational Reasoning

This category evaluates whether the models are capable of interpreting fine-grained, domain-specific spatial relations within molecular complexes, such as hydrogen bonds between atom pairs, and then reasoning about how to manipulate the visualization to highlight specific interactions.

Interaction Location (Multiple-Choice) The model is provided with six orthographic projections of a molecular complex, along with a specified atom pair representing a hydrogen bond (e.g., ARG 45 NH2 A, O1B, denoting the interaction between the NH2 atom of residue ARG45 on chain A and the O1B atom on the ligand). The objective is to distinguish the correct transformation that repositions the corresponding interaction to the center of the visualization from three other decoys. The decoy transformations are generated by perturbing the sign and magnitude of the ground-truth translation parameters.

###### 4.2.3. Global Relational Reasoning

This series of tasks considers the overall spatial relations of the entire complex instead of single localized ones, which is inherently more difficult than previous tasks, as they test the ability of the models to reason high-order combinations of spatial operations.

Ligand Docking (Cloze) This task emulates the molecular docking process to evaluate whether the model can infer the complementary binding configuration and corresponding geometric transformations. The model is provided with three orthographic views of the ligand alone, the pocket alone, and one undocked complex view obtained by translating and rotating the ligand away from its pocket. Rotation angles are uniformly sampled from [−90◦,90◦], while translation distances are adaptively determined for each complex to minimize spatial overlap between the displaced ligand and pocket (Specific details can be found in Appendix A). The model must predict the sequence of transformations required to recover the native docking conformation, such as roll y 45, move x -12. Each complex generates six training samples (21,018 in total) and three test samples (1,470 in total).

Pocket-Ligand Interaction (Cloze) This task extends the Residue–Ligand Interaction task to the entire binding pocket, requiring the model to reason about global intermolecular contact patterns. Given six orthographic projections of the full complex, the models are required to output all hydrogen-bond interactions between the ligand and the pocket in a structured format like ARG 221 NH2 A, O22; ARG 221 N A, O23; ARG 221 NE A, O22. Each interaction is expressed as a tuple specifying the residue type, residue index, interacting atom in the residue, chain identity, and interacting atom in the ligand, with semicolons separating multiple entries.

##### 5. Experiments

We first introduce the experimental setup in Sec. 5.1, and then report the main results of all compared models on our benchmark in Sec. 5.2. Finally, we conduct factor analysis and case study in Sec. 5.3.

###### 5.1. Setup

Benchmark Subsets Due to the expensive cost of closed-source models (especially reasoning models), we create MiSI-Bench(tiny) by randomly sampling 50 question-answer pairs from each task in our dataset. This tiny subset will be used for evaluating the performance of all closed-source models and open-source mixture-of-experts (MoE) models, ensuring an intuitive and fair comparison.

All models are evaluated under few-shot settings to provide them with necessary scientific prior knowledge.

Metrics For Multiple-Choice Questions and Zooming task, we follow the convention to adopt Accuracy (ACC) as the main metric [19, 48], which is calculated as the proportaion of answers exactly matching the ground truth. For Cloze Questions, where answers might involve continuous numerical values and multiple entries, we employ a weighted composite score, as inspired by previous literature [17, 29], to reflect the degree of correctness beyond exact matching. For tasks involving spatial transformations (i.e., move and roll), when the model predicts the correct axis, we will further assign scores based on the predicted values. Specifically, the score is determined by the normalized absolute error between the predicted (dˆ) and groundtruth (d) magnitudes (i.e., distances and angles), |dˆ−d|. For composite tasks involving multiple transformations, such as Ligand Docking, each component operation (e.g., move and roll) contributes equally to the total score, summing up to 1.0. In this task, since the axis for the move operation is fixed to x, scores are assigned only when the sign of the predicted (dˆ) matches the sign of the ground-truth (d). For Residue–Ligand Interaction and Pocket–Ligand Interaction tasks, we compute the ratio of correctly predicted interactions among all provided outputs. To penalize cheating behaviors where models output all the correct interactions along with hallucinations of irrelevant ones, such cases are assigned with a score of 0.5. Furthermore, if the number of hydrogen bonds in the model’s response exceeds twice the number in the ground-truth, we consider the model to be attempting to score through exhaustive enumeration, in which case the model receives a score of 0. Furthermore, we also provide the results for exact matching for Cloze Questions in Appendix B.

Benchmark Models We comprehensively evaluate ten VLMs spanning four major model families, including nine closed-source and one open-source representatives. From Open AI, we include GPT-5-mini (w/ mixed modes of reasoning) [32], o4-mini (w/ reasoning) [33], o3 (w/ reasoning) [33], and GPT-4.1 (w/o reasoning) [31]. From Anthropic’s Claude series, we test Claude 4.5 Sonnet (w/ reasoning) [5], Claude 4 Opus (w/ reasoning) [4], and Claude 3.5 Sonnet (w/o reasoning) [3]. From Google’s Gemini series, we include Gemini-2.5-pro (w/ reasoning) [14] and Gemini-2.5-flash-lite (w/ reasoning) [13]. For open-source models, our preliminary experiments show that most models below 32B parameters achieve very low performance on MiSI-Bench. Therefore, we select the strong Qwen3-vl235b-a22b-thinking (w/ reasoning) [36] from larger opensource models as a representative baseline. Furthermore, we propose Qwen2.5VL-7B-SFT, which is finetuned on the

Table 1. Evaluation on MiSI-Bench. Bold indicates the best result among all models. Since in the Trans-Rot. and Rot-Rot. tasks, the predictions of almost all models are close to random guessing, we exclude these two rows when calculating the average scores.

Res-Lig Inter Pos. Res-Lig Inter Neg.

Inter Location Poc-Lig Inter.

Translation

Trans-Rot. Rot-Rot. Docking

Rotation Zooming

Methods Rank Avg. Unit Task Composite Task Human Level - 81.18 100.00 70.18 30.00 100.00 100.00 32.00 26.00 74.54 92.00 82.78 Reasoning Models GPT-5-mini 9 27.71 47.71 30.55 4.00 29.33 34.00 28.00 22.44 27.24 47.82 1.01 O4-mini 8 28.55 39.71 36.36 2.08 12.67 76.00 40.00 20.00 31.51 30.00 0.00 O3 3 33.65 52.29 43.82 2.00 18.67 94.00 22.00 22.44 20.69 36.00 1.71 Gemini-2.5-pro 6 29.94 50.15 38.88 0.00 28.67 52.00 30.61 21.62 30.38 38.00 1.44 Gemini-2.5-flash-lite 11 16.00 36.29 22.55 4.00 6.67 0.00 30.00 25.00 32.25 26.00 0.25 Claude Opus4 4 33.13 57.43 24.73 6.00 33.67 74.00 12.00 26.00 34.39 34.00 0.77 Claude Sonnet4.5 2 34.37 45.71 44.18 6.00 22.33 84.00 28.00 26.00 34.12 38.00 0.60 Qwen3-vl-235b-a22b-thinking 10 23.34 46.36 25.21 6.00 17.03 25.00 20.40 22.00 29.32 38.00 0.00 General Models GPT-41 7 29.20 29.71 37.45 2.00 7.33 80.00 33.33 29.26 32.90 36.00 0.2 Claude Sonnet3.5 5 31.23 47.14 37.11 10.00 27.50 70.00 18.00 32.00 27.52 28.00 2.55 Our Model Qwen2.5VL-7B-SFT 1 62.96 99.84 99.71 27.14 63.46 89.52 88.44 89.59 24.94 88.37 10.72

training split of our benchmark to investigate how to better trigger the Microscopic Spatial Intelligence in VLMs.

Human-Level Performance We estimate the performance of humans by recruiting PhD candidates in biology to complete the questions for residue-ligand interaction, ligand docking, and pocket-ligand interaction, which requires more domain expertise than the other tasks. For the rest of tasks, we employ PhD candidates in broader field of science, technology, engineering, and mathematics to propose answers. Each participant is required to answer questions in MiSI-Bench(tiny) independently, whose responses are later assessed with the same metrics as the models.

###### 5.2. Main Results

Human Level Performance. The evaluation results are shown in Tab. 1. Human evaluators perform well in most unit tasks, demonstrating strong 3D spatial modeling abilities and the potential to integrate biological knowledge with spatial reasoning for basic interactive tasks. However, their performance decline significantly in complex spatial reasoning tasks. They manage small-angle rotations by tracking key atomic changes, while large-scale rotations—requiring maintained spatial continuity and multiatom tracking—increase cognitive load and impaired axis and angle judgment. Zooming tasks prove even more challenging, as their judgments rely on overall intuition regard-

ing boundary shifts and atomic density changes, lacking clear reference points and resulting in greater estimation errors.

In composite tasks, consecutive spatial operations (e.g., Trans-Rot., Rot-Rot.) lead to error accumulation and frequent reference frame shifts, significantly degrading human performance. Such tasks impose high demands on working memory and the stability of spatial mental simulation, forming a bottleneck in human performance. In Docking task, performance is the poorest due to the need for both sequential spatial transformations and biological knowledge to determine hydrogen bond formation and optimal docking positions. For Poc-Lig Inter. task, the main challenge lies in integrating multiple 2D views to reconstruct the 3D conformation of occluded residues before identifying hydrogen bonds, making it highly demanding. In contrast, Inter Location. task are simpler: they involve no rotational operations—only distance perception—and provide hydrogen bond information upfront, eliminating the need for specialized knowledge.

Advancing VLMs Performance. As evidenced by the results in the table, all advanced models perform suboptimally across various tasks in MiSI-Bench(tiny). Overall, the models exhibit better performance in distancerelated tasks compared to rotation-related ones. For instance, in tasks such as ”Translation” versus ”Rotation”, and ”Interaction Location” versus ”Rotation–Rotation Move-

ment,” the models consistently achieve higher scores in the former. This may stem from the fact that most current VLMs are primarily trained on two-dimensional data, making distance—as a two-dimensional attribute—more readily adaptable for the models. Furthermore, in the ”Residue–Ligand Interaction–Pos” and ”Pocket–Ligand Interaction” tasks, the performance gap between the models and human-level performance is most pronounced, highlighting their still-insufficient knowledge reserves in specialized domains such as biology. In contrast, in the ”Residue–Ligand Interaction–Neg” task, most models perform relatively well, likely because the greater spatial distance between residues and ligands in negative samples allow the models to make correct judgments based solely on spatial proximity.

SFT Model Performance. Experimental results demonstrate that after fine-tuning on the MiSI-Bench dataset, model performance improves significantly, surpassing mainstream VLMs across all tasks and exceeding humanlevel performance in complex spatial tasks such as Rotation. Notably, in the Rot–Rot. task, where human performance approaches random guessing, the model maintains approximately 90% accuracy, indicating that advanced VLMs possess potential for 3D spatial cognition. Previous underperformance of advancing VLMs may have stemmed from domain adaptation barriers: although models have generic spatial understanding, they lack visual priors for specialized structures such as proteins, hindering knowledge transfer. Appropriate fine-tuning can establish crossdomain mappings and unlock their spatial reasoning capabilities. However, in tasks such as Res/Poc–Lig Inter., which rely on domain-specific knowledge, models still lag behind humans, suggesting that the absence of domain priors in foundational training remains a bottleneck. Future work should focus on further exploring the spatial potential of models and investigating how to effectively integrate explicit knowledge from scientific fields such as structural biology.

###### 5.3. Analysis

Factor Analysis. In this section, we conduct a detailed analysis of the suboptimal performance exhibited by the SFT model on the Res-Lig Inter Pos. and Zooming tasks. Fig. 4(a) and (b) present the prediction accuracy of the model across different statistical intervals for these two tasks. As shown in Fig. 4(a), the prediction accuracy decreases sharply as the number of hydrogen bonds increases, indicating that the model struggles to identify all hydrogen bonds in scenarios with complex hydrogen-bonding interactions. In Fig. 4(b), the prediction error rate curve exhibits an initial increase followed by a decrease. We hypothesize that this pattern may stem from the model’s failure to generalize uniformly across the entire scale space. The observed

(A) Exact Match Accuracy by Number of Hydrogen Bonds

100

ExactMatchAccuracy(%)

89.52%

80

64.72%

60

n = 1746

40

n = 1094 28.70%

20.51% n = 39

20

n = 223

0

0 (No H-bond)

1 2 3

Number of Hydrogen Bonds

###### (B) Error Rate Analysis by Zooming Value

100

70

Highest 86.5%

Error Rate

60

Mean: 72.5%

80

###### NumberofSamples

Sample Count

50

ErrorRate(%)

60

40

30

Lowest 42.9%

40

20

20

10

0

0

40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 Zooming Value

- Figure 4. Factor Analysis of MiSI-Bench.

peak likely corresponds to a visually critical scale in molecular structures, where discriminative structural information is minimal, thereby reducing the parsing efficiency of the model’s attention mechanism.

[Figure 65]

- Figure 5. Case study of the Rotation task.

Case Study. In this section, we examine Claude Sonnet 4.5, the top-performing model among advancing VLMs, by analyzing its reasoning process in failure cases from the Rotation task. As shown in Fig. 5, the model demonstrates a logically sound approach: it identifies conserved residues across structural changes as anchors and infers the rotation axis and angle accordingly. However, in the key region it localizes, although the model correctly detects spatial rearrangements of residues 221 and 225, it misinterprets the

change as ”moved slightly backward,” leading to an incorrect conclusion. In fact, simply observing the positional shift of residue 221 would suggest a rotation around the yaxis. This case indicates that advancing VLMs still lack adequate spatial reasoning capabilities and require more effective mechanisms to elicit such skills.

##### 6. Conclusion

In this work, we establish Microscopic Spatial Intelligence (MiSI) as a distinct and critical challenge for VisionLanguage Models (VLMs), extending beyond macroscopic understanding to the atom-level reasoning essential for scientific discovery. We propose a systematic benchmark framework dubbed as MiSI-Bench, for valuating vari-

- ous advanced VLMs for MiSI. The experiments reveals a significant performance gap between state-of-the-art VLMs and human expertise. Yet, the strong performance of a fine-tuned 7B model underscores the substantial potential of VLMs to master complex spatial transformations, even surpassing humans on complex spatial tasks such as rotation. Ultimately, achieving robust MiSI will require not only scaling model architectures but also the explicit integration of scientific knowledge for real-world scientific applications. References

- [1] Rebecca F Alford, Andrew Leaver-Fay, Jeliazko R Jeliazkov, Matthew J O’Meara, Frank P DiMaio, Hahnbeom Park, Maxim V Shapovalov, P Douglas Renfrew, Vikram K Mulligan, Kalli Kappel, et al. The rosetta all-atom energy function for macromolecular modeling and design. Journal of chemical theory and computation, 13(6):3031–3048, 2017. 3
- [2] Amy C Anderson. The process of structure-based drug design. Chemistry & biology, 10(9):787–797, 2003. 3
- [3] Anthropic. Claude 3.5 sonnet. https://www. anthropic . com / news / claude - 3 - 5 - sonnet,

2025. 6

- [4] Anthropic. Introducing claude 4. https://www. anthropic.com/news/claude-4, 2025. 6
- [5] Anthropic. Introducing claude sonnet 4.5. https:// www.anthropic.com/news/claude-sonnet-45, 2025. 2, 6
- [6] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025. 1
- [7] Xiao-Chen Bai, Greg McMullan, and Sjors HW Scheres. How cryo-em is revolutionizing structural biology. Trends in biochemical sciences, 40(1):49–57, 2015. 3
- [8] Daniil A Boiko, Robert MacKnight, Ben Kline, and Gabe Gomes. Autonomous chemical research with large language models. Nature, 624(7992):570–578, 2023. 1
- [9] Michael M Bronstein, Joan Bruna, Yann LeCun, Arthur Szlam, and Pierre Vandergheynst. Geometric deep learning: going beyond euclidean data. IEEE Signal Processing Magazine, 34(4):18–42, 2017. 3

- [10] Ingrid Carlbom and Joseph Paciorek. Planar geometric projections and viewing transformations. ACM Computing Surveys (CSUR), 10(4):465–502, 1978. 3
- [11] Boyuan Chen, Zhuo Xu, Sean Kirmani, Brain Ichter, Dorsa Sadigh, Leonidas Guibas, and Fei Xia. Spatialvlm: Endowing vision-language models with spatial reasoning capabilities. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14455–14465,

2024. 1

- [12] Vadim Cherezov, Daniel M Rosenbaum, Michael A Hanson, Søren GF Rasmussen, Foon Sun Thian, Tong Sun Kobilka, Hee-Jung Choi, Peter Kuhn, William I Weis, Brian K Kobilka, et al. High-resolution crystal structure of an engineered human β2-adrenergic g protein–coupled receptor. science, 318(5854):1258–1265, 2007. 3
- [13] Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025. 6
- [14] Google DeepMind. Gemini 2.5: Our most intelligent ai model. https://blog.google/technology/ google- deepmind/gemini- model- thinkingupdates-march-2025/, 2025. 6
- [15] Warren L DeLano et al. Pymol: An open-source molecular graphics tool. CCP4 Newsl. protein crystallogr, 40(1):82– 92, 2002. 1
- [16] Simon Eidelman, KG Hayes, KA ea Olive, M AguilarBenitez, C Amsler, D Asner, KS Babu, RM Barnett, J Beringer, PR Burchat, et al. Review of particle physics. Physics letters B, 592(1-4):1–5, 2004. 1, 3
- [17] Mark Everingham, Luc Van Gool, Christopher KI Williams, John Winn, and Andrew Zisserman. The pascal visual object classes (voc) challenge. International journal of computer vision, 88(2):303–338, 2010. 6
- [18] Kaituo Feng, Manyuan Zhang, Hongyu Li, Kaixuan Fan, Shuang Chen, Yilei Jiang, Dian Zheng, Peiwen Sun, Yiyuan Zhang, Haoze Sun, et al. Onethinker: All-inone reasoning model for image and video. arXiv preprint arXiv:2512.03043, 2025. 1
- [19] Chaoyou Fu, Yuhan Dai, Yongdong Luo, Lei Li, Shuhuai Ren, Renrui Zhang, Zihan Wang, Chenyu Zhou, Yunhang Shen, Mengdan Zhang, et al. Video-mme: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 24108–24118, 2025. 6
- [20] Xingyu Fu, Yushi Hu, Bangzheng Li, Yu Feng, Haoyu Wang, Xudong Lin, Dan Roth, Noah A Smith, Wei-Chiu Ma, and Ranjay Krishna. Blink: Multimodal large language models can see but not perceive. In European Conference on Computer Vision, pages 148–166. Springer, 2024. 2
- [21] Bowen Gao, Bo Qiang, Haichuan Tan, Yinjun Jia, Minsi Ren, Minsi Lu, Jingjing Liu, Wei-Ying Ma, and Yanyan Lan. Drugclip: Contrastive protein-molecule representation learning for virtual screening. Advances in Neural Information Processing Systems, 36:44595–44614, 2023. 3

- [22] Wenbing Huang, Rui Jiao, Xiangzhe Kong, Li Zhang, Ziyang Yu, Fangyuan Ren, Wenjuan Tan, and Yang Liu. An equivariant pretrained transformer for unified 3d molecular representation learning. 2025. 3
- [23] Xiangzhe Kong, Wenbing Huang, and Yang Liu. Conditional antibody design as 3d equivariant graph translation. In The Eleventh International Conference on Learning Representations, 2023. 3
- [24] Xiangzhe Kong, Wenbing Huang, and Yang Liu. Generalist equivariant transformer towards 3d molecular interaction learning. In International Conference on Machine Learning, pages 25149–25175. PMLR, 2024. 3
- [25] Xiangzhe Kong, Rui Jiao, Haowei Lin, Ruihan Guo, Wenbing Huang, Wei-Ying Ma, Zihua Wang, Yang Liu, and Jianzhu Ma. Peptide design through binding interface mimicry with pepmimic. Nature biomedical engineering, pages 1–16, 2025. 3
- [26] Xiangzhe Kong, Zishen Zhang, Ziting Zhang, Rui Jiao, Jianzhu Ma, Wenbing Huang, Kai Liu, and Yang Liu. Unimomo: Unified generative modeling of 3d molecules for de novo binder design. In Forty-second International Conference on Machine Learning, 2025. 3
- [27] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Peiyuan Zhang, Yanwei Li, Ziwei Liu, et al. Llava-onevision: Easy visual task transfer. arXiv preprint arXiv:2408.03326, 2024. 1
- [28] Feng Li, Renrui Zhang, Hao Zhang, Yuanhan Zhang, Bo Li, Wei Li, Zejun Ma, and Chunyuan Li. Llava-next-interleave: Tackling multi-image, video, and 3d in large multimodal models. arXiv preprint arXiv:2407.07895, 2024. 1
- [29] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Doll´ar, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In European conference on computer vision, pages 740–755. Springer, 2014. 6
- [30] Andres M. Bran, Sam Cox, Oliver Schilter, Carlo Baldassari, Andrew D White, and Philippe Schwaller. Augmenting large language models with chemistry tools. Nature Machine Intelligence, 6(5):525–535, 2024. 1
- [31] OpenAI. Introducing gpt-4.1 in the api. https:// openai.com/index/gpt-4-1/, 2025. 6
- [32] OpenAI. Introducing gpt-5. https://openai.com/ index/introducing-gpt-5/, 2025. 6
- [33] OpenAI. Introducing o3 and o4 mini. https://openai. com/index/introducing- o3- and- o4- mini/,

2025. 2, 6

- [34] Eric F Pettersen, Thomas D Goddard, Conrad C Huang, Elaine C Meng, Gregory S Couch, Tristan I Croll, John H Morris, and Thomas E Ferrin. Ucsf chimerax: Structure visualization for researchers, educators, and developers. Protein science, 30(1):70–82, 2021. 1, 3
- [35] Pedro O Pinheiro, Arian Rokkum Jamasb, Omar Mahmood, Vishnu Sresht, and Saeed Saremi. Structure-based drug design by denoising voxel grids. In International Conference on Machine Learning, pages 40795–40812. PMLR, 2024. 3
- [36] QwenTeam. Qwen3-vl: Sharper vision, deeper thought, broader action. https://qwen.ai/blog?id=

99f0335c4ad9ff6153e517418d48535ab6d8afef& from=research.latest- advancements- list,

2025. 6

- [37] Arijit Ray, Jiafei Duan, Reuben Tan, Dina Bashkirova, Rose Hendrix, Kiana Ehsani, Aniruddha Kembhavi, Bryan A Plummer, Ranjay Krishna, Kuo-Hao Zeng, et al. Sat: Spatial aptitude training for multimodal language models. arXiv e-prints, pages arXiv–2412, 2024. 2
- [38] Joost Schymkowitz, Jesper Borg, Francois Stricher, Robby Nys, Frederic Rousseau, and Luis Serrano. The foldx web server: an online force field. Nucleic acids research, 33 (suppl 2):W382–W388, 2005. 3

- [39] Kyle Swanson, Wesley Wu, Nash L Bulaong, John E Pak, and James Zou. The virtual lab of ai agents designs new sars-cov-2 nanobodies. Nature, pages 1–3, 2025. 1
- [40] Kexian Tang, Junyao Gao, Yanhong Zeng, Haodong Duan, Yanan Sun, Zhening Xing, Wenran Liu, Kaifeng Lyu, and Kai Chen. Lego-puzzles: How good are mllms at multi-step spatial reasoning? arXiv preprint arXiv:2503.19990, 2025. 3
- [41] Philipp Th¨olke and Gianni De Fabritiis. Equivariant transformers for neural network based molecular potentials. In International Conference on Learning Representations, 2022. 3
- [42] Raphael John Lamarre Townshend, Martin V¨ogele, Patricia Adriana Suriana, Alexander Derry, Alexander Powers, Yianni Laloudakis, Sidhika Balachandar, Bowen Jing, Brandon M. Anderson, Stephan Eismann, Risi Kondor, Russ Altman, and Ron O. Dror. ATOM3d: Tasks on molecules in three dimensions. In Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 1), 2021. 3
- [43] Fei Wang, Xingyu Fu, James Y Huang, Zekun Li, Qin Liu, Xiaogeng Liu, Mingyu Derek Ma, Nan Xu, Wenxuan Zhou, Kai Zhang, et al. Muirbench: A comprehensive benchmark for robust multi-image understanding. arXiv preprint arXiv:2406.09411, 2024. 2
- [44] Renxiao Wang, Xueliang Fang, Yipin Lu, Chao-Yie Yang, and Shaomeng Wang. The pdbbind database: methodologies and updates. Journal of medicinal chemistry, 48(12):4111– 4119, 2005. 2, 3, 1
- [45] Junfei Wu, Jian Guan, Kaituo Feng, Qiang Liu, Shu Wu, Liang Wang, Wei Wu, and Tieniu Tan. Reinforcing spatial reasoning in vision-language models with interwoven thinking and visual drawing. arXiv preprint arXiv:2506.09965,

2025. 1

- [46] Jihan Yang, Shusheng Yang, Anjali W Gupta, Rilyn Han, Li Fei-Fei, and Saining Xie. Thinking in space: How multimodal large language models see, remember, and recall spaces. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 10632–10643, 2025. 1, 2
- [47] Baiqiao Yin, Qineng Wang, Pingyue Zhang, Jianshu Zhang, Kangrui Wang, Zihan Wang, Jieyu Zhang, Keshigeyan Chandrasegaran, Han Liu, Ranjay Krishna, et al. Spatial mental modeling from limited views. In Structural Priors for Vision Workshop at ICCV’25, 2025. 1
- [48] Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming

- Ren, Yuxuan Sun, et al. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9556– 9567, 2024. 6
- [49] Jiahui Zhang, Yurui Chen, Yanpeng Zhou, Yueming Xu, Ze Huang, Jilin Mei, Junhui Chen, Yu-Jie Yuan, Xinyue Cai, Guowei Huang, et al. From flatland to space: Teaching vision-language models to perceive and reason in 3d. arXiv preprint arXiv:2503.22976, 2025. 1
- [50] Baining Zhao, Ziyou Wang, Jianjie Fang, Chen Gao, Fanhang Man, Jinqiang Cui, Xin Wang, Xinlei Chen, Yong Li, and Wenwu Zhu. Embodied-r: Collaborative framework for activating embodied spatial reasoning in foundation models via reinforcement learning. In Proceedings of the 33rd ACM International Conference on Multimedia, pages 11071–11080, 2025. 1
- [51] Yiming Zuo, Karhan Kayan, Maggie Wang, Kevin Jeon, Jia Deng, and Thomas L Griffiths. Towards foundation models for 3d vision: How close are we? In 2025 International Conference on 3D Vision (3DV), pages 1285–1296. IEEE,

2025. 2

##### A. Dataset Details

In this section, we present additional details regarding the construction of the dataset.

###### A.1. Data Generation Pipeline

As shown in Fig. 6, our dataset construction process consists of three main steps.

The first step is Dataset Collection. We download the PDBBind dataset [44], retaining only the ligand and pocket (collectively referred to as the complex) for each sample. As the first exploratory dataset for microscopic spatial intelligence, we reduce the overall complexity by removing the solvent and hiding the hydrogen. Subsequently, we color all oxygen, nitrogen, and carbon atoms red, blue, and gray, respectively, in accordance with common coloring standards in the field. Notably, for pocket residues, we apply alternating yellow and purple coloring to facilitate the model’s ability to distinguish adjacent residues in subsequent interaction (hydrogen bond) recognition tasks.

The second step is Annotation. For each complex, we use the ChimeraX [34] command get sel screen to obtain the screen coordinates of all atoms. We then employ ChimeraX’s built-in hydrogen bond calculation function to identify all hydrogen bonds between the pocket and ligand. All this information is annotated for every complex and stored for direct use in constructing subsequent subtaskspecific datasets.

The third step is Subtask-Specific Data Generation. For each subtask, we first develop corresponding code according to its requirements. We then use ChimeraX to generate multiple image samples for every complex in the training and test sets. Finally, we design question-answering (QA) templates for each subtask and populate them with the metainformation of each sample to form complete data instances.

###### A.2. Zooming

For the zooming task, we compute the probability density function of the movement depth required to translate all interactions from their initial states to the screen center in the training set, as shown in Fig. 7. The results indicate that most movement depths fall within the range of 20–80 angstroms (A)).˚ Through empirical testing, we observe that when the movement depth lies in the 20–40 Ainterval,˚ a one-unit difference in the zooming operation (e.g., move z 25 vs. move z 26) produces only minor changes in the output, making them difficult to distinguish and thus unsuitable as training data (as illustrated in Fig. 8). On the other hand, for movement depths in the 60–80 Arange,˚ due to the varying spatial conformations of different PDB structures in their initial states, some structures become zoomed in to the extent that only a single atom remains visible when depth values exceed 60 A.˚ Such samples are likewise inadequate for training the model to discern specific zooming

depth values. Therefore, we exclusively select depth values within the 40–60 Ainterval˚ to generate our dataset.

###### A.3. Residue-Ligand Interaction

In tasks related to interaction, the calculation of groundtruth hydrogen bonds follows the default protocol of the software ChimeraX [34]. The distance and angle cutoffs for hydrogen bonding are based on a survey of small-molecule crystal structures, as described in ChimeraX. Additionally, the option to relax distance and angle criteria—that is, whether to incorporate tolerance values beyond the precise criteria for identifying hydrogen bonds (which involve several distinct distance and angle thresholds depending on the atom types involved)—is also adopted from the reference provided in ChimeraX. Specifically, a distance tolerance of 0.4 Aand˚ an angle tolerance of 20 degrees are used.

###### A.4. Ligand Docking

In ligand docking task, to enable the model to better observe the respective conformational and geometric information of the protein pocket and the displaced ligand, it is essential to minimize the overlapping region between the displaced ligand and the pocket. In ChimeraX, most molecular complexes initially occupy the full vertical extent (e.g., the Yaxis) of the screen; therefore, we only consider translating the ligand horizontally (e.g., along the X-axis) to either the far left or far right side of the screen. We randomly select the first complex (PDB ID: 1ugx) from the training dataset as a reference. In its native docking conformation, the mean screen coordinates of all atoms in this complex are denoted as (Xbasec ,Ybasec ,Zbasec ). When the ligand alone is moved to the far right side of the screen, the mean screen coordinates of all atoms in the ligand become (Xbasel ,Ybasel ,Zbasel ).

For other complexes in the dataset, the mean screen coordinates of all atoms in each complex under the native docking conformation are obtained as (xc,yc,zc). Similarly, under the same conformation, the maximum and minimum X-axis screen coordinates of all atoms in the ligand are denoted as (xlmax,xlmin). Based on this, the distance required to move to the farthest point is approximated according to the Field of View. Specifically, the distance to move to the far-right is calculated as dstr = −zc ∗ Xbasel /Zbasec −xlmax, and the distance to move to the far-left is dstl = −(zc ∗ Xbasel /Zbasec − xlmax). Subsequently, numerical values are randomly sampled from [0,1,2] and subtracted from or added to dstr and dstl, respectively. These adjusted distances are then combined with subsequent rotation operations to generate additional dataset samples.

##### B. More Results

In this section, we present the Exact Matching Accuracy of all evaluated models and human evaluators on the MiSI-Bench dataset. A prediction is considered correct

[Figure 66]

|[Figure 67]|
|---|

|[Figure 68]<br><br>[Figure 69]<br><br>Program Run by Chimerax|
|---|

[Figure 70]

- -Only download pocket and ligand
- -Remove the solvent
- -Hide the hydrogen

[Figure 71]

[Figure 72]

- -O atom - red
- -N atom - blue
- -C atom - grey
- -Residues - alternating
- -Lable the residues
- -Lable the ligand atoms

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

|[Figure 77]<br><br>[Figure 78]<br><br>[Figure 79]<br><br>[Figure 80]<br><br>[Figure 81]<br><br>[Figure 82]<br><br>Three-view generation<br><br>|[Figure 83]<br><br>[Figure 84]|
|---|
<br><br>The changed main view|
|---|

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

Record

[Figure 93]

[Figure 94]

- -Hydrogen bond information in the scene
- -The coordinates of all atoms

[Figure 95]

Problem templates one unique template for each task

Dataset collection Annotated Technology Roadmap

- Figure 6. Data generation pipeline. Our pipeline comprises three key stages: data collection and filtering, data annotation, and a unified

module for synthesizing both images and question-answer pairs using Chimera with fixed templates. Table 2. Evaluation on MiSI-Bench. We employ Exact Matching Accuracy as the evaluation metric. Bold indicates the best result among all models. Since in the Trans-Rot. and Rot-Rot. tasks, the predictions of almost all models are close to random guessing, we exclude these two rows when calculating the average scores.

Res-Lig Inter Pos. Res-Lig Inter Neg.

Inter Location Poc-Lig Inter.

Translation

Trans-Rot. Rot-Rot. Docking

Rotation Zooming

Methods Rank Avg. Unit Task Composite Task Human Level - 76.25 100.00 58.00 30.00 100.00 100.00 32.00 26.00 60.00 92.00 70.00 Reasoning Models GPT-5-mini 8 16.23 10.00 10.00 4.00 22.00 34.00 28.00 22.44 2.04 47.82 0.00 O4-mini 9 13.01 6.00 12.00 2.08 8.00 76.00 40.00 20.00 0.00 30.00 0.00 O3 2 33.65 22.00 20.00 2.00 12.00 94.00 22.00 22.44 0.00 36.00 0.00 Gemini-2.5-pro 6 17.54 14.89 23.40 0.00 12.00 52.00 30.61 21.62 0.00 38.00 0.00 Gemini-2.5-flash-lite 11 9.04 36.29 2.00 4.00 4.00 0.00 30.00 25.00 0.00 26.00 0.00 Claude Opus4 4 19.54 14.29 6.00 6.00 22.00 74.00 12.00 26.00 0.00 34.00 0.00 Claude Sonnet4.5 3 20.75 8.00 14.00 6.00 14.00 84.00 28.00 26.00 2.00 38.00 0.00 Qwen3-vl-235b-a22b-thinking 10 11.80 14.29 4.55 6.00 6.52 25.00 20.40 22.00 0.00 38.00 0.00 General Models GPT-41 7 17.50 4.00 16.00 2.00 2.00 80.00 33.33 29.26 0.00 36.00 0.00 Claude Sonnet3.5 5 18.02 10.00 8.16 10.00 18.00 70.00 18.00 32.00 0.00 28.00 0.00 Our Model Qwen2.5VL-7B-SFT 1 57.56 98.88 97.48 27.14 57.52 89.52 88.44 89.59 0.75 88.37 0.78

[Figure 96]

- Figure 7. The probability density function of the movement depth.

[Figure 97]

[Figure 98]

Figure 8. Comparison of different movement depths.

only if it exactly matches the ground-truth. The corresponding results are shown in Table Tab. 2. It should be noted that for the Protein–Ligand Interaction task, while the original test set contains a certain number of complexes with-

- out hydrogen bonds, there is only one such complex in MiSI-Bench(tiny). Given that predicting the absence of hydrogen bonds is considerably simpler than predicting all hydrogen bonds correctly, we report the Exact Matching Accuracy of the models only for complexes that contain hydrogen bonds.

From the table, we can observe that when Exact Matching Accuracy is applied, the performance gap between all advancing VLMs and our SFT model becomes more pronounced. Particularly in the Translation and Rotation tasks, the transformation of the metric have a limited impact on the SFT model, whereas the performance of advancing VLMs exhibits a substantial decline. This indicates that while advancing VLMs possess strong potential for spatial understanding and reasoning, effective methods are required to activate this capability. For the SFT model, the performance gap with human evaluators is further widened in tasks related to interaction recgnition. This indicates that current models lack specialized domain knowledge, and suggests that incorporating such knowledge during the pre-training phase may be necessary for progressing toward more general artificial intelligence.

##### C. Visualization

In this section, we present complete examples for all nine tasks, as described in Figs. 9 to 17, respectively. For display purposes, the background color of the images has been changed from black to transparent. The highlighted content in each question represents unique information for that specific sample, while the remaining text in black constitutes the unified prompt for the subtask.

[Figure 103]

###### Answer

move x 4

Front after transformation

Problem

I will provide you with four images showing the binding interaction between a protein (PDB ID: 5ey0) and a small molecule ligand. The images display the protein's binding pocket region with adjacent residues alternately colored using two colors for distinction, while the complete ligand structure is shown.

## Image Description:

- - **Image 1**: Front view (main perspective)
- - **Image 2**: Left view (viewed from the left side)
- - **Image 3**: Top view (viewed from above)
- - **Image 4**: Front view after spatial transformation (the pocket-ligand complex has been transformed in 3D space and then observed from the front view perspective) ## Visualization Details:
- - **General Rules**:
- - Hydrogen atoms are not displayed
- - Water molecules (H2O) are not displayed
- - **Protein's binding pocket region**: All atoms of all constituent residues are displayed
- - Carbon atoms can be colored using either #BF99F2 (purple) or #F2B366 (orange)
- - The residue is numbered and labeled in the image
- - **Ligand**: Small molecule with all atoms displayed
- - Carbon atoms: Gray color
- - Each atom is labeled with its atom name
- - **Common Atom Colors** (both pocket and ligand):
- - Oxygen atoms: Red
- - Nitrogen atoms: Blue

## Coordinate System Definition (based on front view):

- - **X-axis**: Horizontal direction, rightward is positive
- - **Y-axis**: Vertical direction, upward is positive

### Translation (Move) Rules:

- - Command format: move <axis> <dst> (translate along the <axis> by <dst> units)
- - **axis**: Must be x or y
- - Positive dst: movement in positive axis direction (rightward for x-axis, upward for y-axis)
- - Negative dst: movement in negative axis direction (leftward for x-axis, downward for y-axis)

## Task: Please analyze the transformation from Image 1 to Image 4 and determine the required translation command to achieve this transformation. ## Analysis Requirements:

- - Compare the overall position changes between Image 1 and Image 4
- - Identify which axis/axes appear to be involved in the translation
- - Estimate translation distances based on visual cues and structural landmarks
- - Use Images 2 and 3 as reference views to help determine the spatial transformations

## Output Format: [**Single** translation command in the exact format: move <axis> <dst>] Examples: move x 1, move y -3

- Figure 9. Sample visualization for the translation task. Zoom in for greater detail.

[Figure 108]

###### Answer

roll y 15

Front after transformation

Problem

I will provide you with four images showing the binding interaction between a protein (PDB ID: 5ey0) and a small molecule ligand. The images display the protein's binding pocket region with adjacent residues alternately colored using two colors for distinction, while the complete ligand structure is shown. ## Image Description:

- - **Image 1**: Front view (main perspective)
- - **Image 2**: Left view (viewed from the left side)
- - **Image 3**: Top view (viewed from above)
- - **Image 4**: Front view after spatial transformation (the pocket-ligand complex has been transformed in 3D space and then observed from the front view perspective)

## Visualization Details:

- - **General Rules**:
- - Hydrogen atoms are not displayed
- - Water molecules (H2O) are not displayed
- - **Protein's binding pocket region**: All atoms of all constituent residues are displayed
- - Carbon atoms can be colored using either #BF99F2 (purple) or #F2B366 (orange)
- - The residue is numbered and labeled in the image
- - **Ligand**: Small molecule with all atoms displayed
- - Carbon atoms: Gray color
- - Each atom is labeled with its atom name
- - **Common Atom Colors** (both pocket and ligand):
- - Oxygen atoms: Red
- - Nitrogen atoms: Blue

## Coordinate System Definition (based on front view):

- - **X-axis**: Horizontal direction, rightward is positive
- - **Y-axis**: Vertical direction, upward is positive
- - **Z-axis**: Perpendicular to screen, outward is positive

## Rotation Rules:

- - Follow the right-hand rule: Four fingers point in the positive rotation direction
- - Command format: roll <axis> <angle> (rotate around the <axis> by <angle> degrees)
- - **axis**: Must be x, y, or z
- - Positive angle: rotation in positive direction (right-hand rule)
- - Negative angle: rotation in negative direction

## Task: Please analyze the transformation from Image 1 to Image 4 and determine the required rotation command to achieve this transformation. ## Analysis Requirements:

- - Compare the overall orientation changes between Image 1 and Image 4
- - Identify which axis/axes appear to be involved in the rotation
- - Estimate rotation angles based on visual cues and structural landmarks
- - Use Images 2 and 3 as reference views to help determine the spatial transformations

## Output Format: [**Single** rotation command in the exact format: roll <axis> <angle>] Examples: roll x 15, roll y -60, roll z 90

- Figure 10. Sample visualization for the rotation task. Zoom in for greater detail.

[Figure 109]

[Figure 110]

[Figure 111]

Left Front Top

[Figure 112]

[Figure 113]

###### Answer

move z 40

Front after transformation

Problem

I will provide you with four images showing the binding interaction between a protein (PDB ID: 5ey0) and a small molecule ligand. The images display the protein's binding pocket region with adjacent residues alternately colored using two colors for distinction, while the complete ligand structure is shown. ## Image Description:

- - **Image 1**: Front view (main perspective)
- - **Image 2**: Left view (viewed from the left side)
- - **Image 3**: Top view (viewed from above)
- - **Image 4**: Front view after spatial transformation (the pocket-ligand complex has been transformed in 3D space and then observed from the front view perspective)

## Visualization Details:

- - **General Rules**:
- - Hydrogen atoms are not displayed
- - Water molecules (H2O) are not displayed
- - **Protein's binding pocket region**: All atoms of all constituent residues are displayed
- - Carbon atoms can be colored using either #BF99F2 (purple) or #F2B366 (orange)
- - The residue is numbered and labeled in the image
- - **Ligand**: Small molecule with all atoms displayed
- - Carbon atoms: Gray color
- - Each atom is labeled with its atom name
- - **Common Atom Colors** (both pocket and ligand):
- - Oxygen atoms: Red
- - Nitrogen atoms: Blue

## Coordinate System Definition (based on front view):

- - **Z-axis**: Perpendicular to screen

### Scaling Rules:

- - Command format: move z <dst> (adjust the scale level based on <dst>)
- - **axis**: Must be z
- - Positive dst: zoom in (atom appears larger)
- - Negative dst: zoom out (atom appears smaller)

## Task: Please analyze the transformation from Image 1 to Image 4 and determine the required scaling command to achieve this transformation. ## Analysis Requirements:

- - Compare the overall changes between Image 1 and Image 4
- - Estimate scale levels based on visual cues and structural landmarks
- - Use Images 2 and 3 as reference views to help determine the spatial transformations

## Output Format: [**Single** scale command in the exact format: move z <dst>] Examples: move z 42

- Figure 11. Sample visualization for the zooming task. Zoom in for greater detail.

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

Left Front Top Answer

[Figure 118]

[Figure 119]

[Figure 120]

Yes: ARG NH2, O1B; ARG NE, O2G

Right Back Bottom

###### Problem

I will provide you with six images showing the binding interaction between a specific protein residue (PDB ID: 5ey0) and a small molecule ligand. All images display the target residue within the binding pocket region, along with the complete ligand, with full atomic details shown for both components. ## Image Description:

- - **Image 1**: Front view (main perspective)
- - **Image 2**: Left view (viewed from the left side)
- - **Image 3**: Top view (viewed from above)
- - **Image 4**: Back view (viewed from behind)
- - **Image 5**: Right view (viewed from the right side)
- - **Image 6**: Bottom view (viewed from below) ## Visualization Details:
- - **General Rules**:
- - Hydrogen atoms are not displayed
- - Water molecules (H2O) are not displayed
- - **Target Residue**: All atoms of the target residue are displayed
- - Carbon atoms can be colored using either #BF99F2 (purple) or #F2B366 (orange)
- - Each atom is labeled with its atom name
- - **Ligand**: Small molecule with all atoms displayed
- - Carbon atoms: Gray color
- - Each atom is labeled with its atom name
- - **Common Atom Colors** (both pocket and ligand):
- - Oxygen atoms: Red
- - Nitrogen atoms: Blue

## Task: Analyze the six images to determine if there are any hydrogen bonds between the target protein residue and the ligand. If hydrogen bonds exist, identify all of them.

## Output Format: Your answer should follow one of these two formats:

- ### Case 1: No hydrogen bonds exist Simply output: **No**
- ### Case 2: Hydrogen bonds exist Output: **Yes: [hydrogen bond list]**

Where the hydrogen bond list follows this format:

**[Amino Acid Type] [Donor/Acceptor Atom], [Ligand Atom Name]** ### Format Rules:

- - Use standard three-letter amino acid codes (e.g., ARG, GLY, SER, etc.)
- - Always list protein residue information first, then ligand information
- - Separate multiple hydrogen bonds with semicolons (;)
- - Examples:
- - "No" (if no hydrogen bonds)
- - "Yes: VAL O, N1" (if one hydrogen bond)
- - "Yes: VAL O, N1; VAL O, N2" (if multiple hydrogen bonds) ## Analysis Requirements:
- - Examine all six views to get complete spatial understanding
- - Identify atoms capable of hydrogen bonding
- - Consider the 3D geometry from multiple perspectives
- - Focus on interactions at typical hydrogen bonding distances
- - First determine if ANY hydrogen bonds exist, then identify specific bonds if present Please analyze the images systematically and provide your answer in the specified format.

- Figure 12. Sample visualization for the residue-ligand interaction task. Zoom in for greater detail.

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

Reference protein

Left Front Top

Transformed

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

A B

[Figure 130]

[Figure 131]

Left Front Top

Target protein

Answer: C

C D

|I will provide you with multiple images showing protein-ligand binding interactions. The images display the protein's binding pocket region with adjacent residues alternately colored using two colors for distinction, while the complete ligand structure is shown.<br><br>## Part 1: Reference Transformation (4 images) These 4 images show a reference protein (PDB ID: 5ey0) and demonstrate a spatial transformation:<br><br>- **Image 1**: Initial front view (main perspective)<br>- **Image 2**: Initial left view (viewed from the left side)<br>- **Image 3**: Initial top view (viewed from above)<br>- **Image 4**: Front view after transformation (one rotation + one translation)<br><br><br>## Part 2: Apply Transformation (7 images) Now, you need to apply the SAME transformation to a NEW protein (PDB ID: 2i6b):<br><br><br>- **Image 5**: Target protein - Initial front view (main perspective)<br>- **Image 6**: Target protein - Initial left view (viewed from the left side)<br>- **Image 7**: Target protein - Initial top view (viewed from above)<br>- **Images 8-11**: Four options (A, B, C, D) showing different possible results from the front view (main perspective) ## Visualization Details:<br>- **General Rules**:<br>- Hydrogen atoms are not displayed<br>- Water molecules (H2O) are not displayed<br>- **Protein's binding pocket region**: All atoms of all constituent residues are displayed<br>- Carbon atoms can be colored using either #BF99F2 (purple) or #F2B366 (orange)<br>- The residue is numbered and labeled in the image<br>- **Ligand**: Small molecule with all atoms displayed<br>- Carbon atoms: Gray color<br>- Each atom is labeled with its atom name<br>- **Common Atom Colors** (both pocket and ligand):<br>- Oxygen atoms: Red<br>- Nitrogen atoms: Blue ## Coordinate System (based on front view):<br>- **X-axis**: Horizontal, rightward is positive<br>- **Y-axis**: Vertical, upward is positive<br>- **Z-axis**: Perpendicular to screen, outward is positive ## Transformation Commands:<br><br>1. **Rotation**: roll <axis> <angle> (rotate around the <axis> by <angle> degrees)<br><br>- Follow right-hand rule (four fingers point in positive rotation direction)<br>- axis: x, y, or z<br>- Positive angle: rotation in positive direction<br>- Negative angle: rotation in negative direction<br><br><br>2. **Translation**: move <axis> <distance> (translate along the <axis> by <dst> units)<br><br><br>- axis: x or y<br>- Positive distance: rightward (x) or upward (y)<br>- Negative distance: leftward (x) or downward (y) ## Task:<br><br><br>1. Analyze the spatial transformation from Images 1-3 to Image 4 in the reference protein<br>2. Identify the specific rotation and translation commands<br>3. Apply the same transformation logic to the target protein (Images 5-7)<br>4. Select which option (A, B, C, or D) correctly shows the result<br><br><br>## Output Format: [Single letter: A, B, C, or D]<br><br>Problem<br><br>[Figure 132]|
|---|

- Figure 13. Sample visualization for the translation-rotation movement task. Zoom in for greater detail.

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

Reference protein

Left Front Top

Transformed

[Figure 137]

[Figure 138]

[Figure 139]

Answer: B

A B

[Figure 140]

[Figure 141]

Problem

C D

I will provide you with multiple images showing the binding interaction between a protein (PDB ID: 5ey0) and a small molecule ligand. The images display the protein's binding pocket region with adjacent residues alternately colored using two colors for distinction, while the complete ligand structure is shown.

- ## Part 1: Reference Transformation (4 images)

- - **Image 1**: Initial front view (main perspective)
- - **Image 2**: Initial left view (viewed from the left side)
- - **Image 3**: Initial top view (viewed from above)
- - **Image 4**: Front view after TWO sequential rotations (rotation1 then rotation2)

- ## Part 2: Apply SAME transformation, and then select the correct answer from the options below (4 images)

- - **Images 5-8**: Four options (A, B, C, D) showing different possible results from the front view (main perspective) ## Visualization Details:
- - **General Rules**:
- - Hydrogen atoms are not displayed
- - Water molecules (H2O) are not displayed
- - **Protein's binding pocket region**: All atoms of all constituent residues are displayed
- - Carbon atoms can be colored using either #BF99F2 (purple) or #F2B366 (orange)
- - The residue is numbered and labeled in the image
- - **Ligand**: Small molecule with all atoms displayed
- - Carbon atoms: Gray color
- - Each atom is labeled with its atom name
- - **Common Atom Colors** (both pocket and ligand):
- - Oxygen atoms: Red
- - Nitrogen atoms: Blue ## Coordinate System (based on front view):
- - **X-axis**: Horizontal, rightward is positive
- - **Y-axis**: Vertical, upward is positive
- - **Z-axis**: Perpendicular to screen, outward is positive ## Rotation Rules:
- - Follow the right-hand rule: Four fingers point in the positive rotation direction
- - Command format: roll <axis> <angle> (rotate around the <axis> by <angle> degrees)
- - **axis**: Must be x, y, or z
- - Positive angle: rotation in positive direction (right-hand rule)
- - Negative angle: rotation in negative direction

## Task: The transformation from Images 1-3 to Image 4 involves TWO sequential rotations. The ORDER of rotations matters! Rotation operations are not commutative. Now, starting from Image 4 (the result after the first two rotations), you need to:

- 1. Analyze the spatial transformation in the reference example (Images 1-3 → Image 4) to deduce the exact two rotation commands
- 2. Apply the same two rotation commands, in the identical sequence, to Image 4 itself
- 3. Select which option (A, B, C, or D) correctly shows the result

## Output Format: [Single letter: A, B, C, or D]

- Figure 14. Sample visualization for the rotation-rotation movement task. Zoom in for greater detail.

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

Left Front Top

[Figure 146]

Protein binding pocket

[Figure 147]

[Figure 148]

Left Front Top

Showing both pocket and displaced ligand together

Ligand

Answer

roll x 75, move x -14

|I will provide you with seven images showing a protein binding pocket and a small molecule ligand (PDB ID: 5ey0). The ligand has been moved and rotated away from its optimal binding position. Your task is to determine the operations needed to dock the ligand back into the binding pocket. ## Image Description:<br><br>- **Images 1-3**: Protein binding pocket only (isolated view)<br>- Image 1: Front view of pocket<br>- Image 2: Left view of pocket<br>- Image 3: Top view of pocket<br>- **Images 4-6**: Ligand only (isolated view at its current displaced position)<br>- Image 4: Front view of displaced ligand<br>- Image 5: Left view of displaced ligand<br>- Image 6: Top view of displaced ligand<br>- Note: These views show the ligand at its current position after being moved and rotated<br>- **Image 7**: Combined view showing both pocket and displaced ligand together (front view perspective) ## Visualization Details:<br>- **General Rules**:<br>- Hydrogen atoms are not displayed<br>- Water molecules (H2O) are not displayed<br>- **Protein's binding pocket region**: All atoms of all constituent residues are displayed<br>- Carbon atoms can be colored using either #BF99F2 (purple) or #F2B366 (orange)<br>- The residue is numbered and labeled in the image<br>- **Ligand**: Small molecule with all atoms displayed<br>- Carbon atoms: Gray color<br>- Each atom is labeled with its atom name<br>- **Common Atom Colors** (both pocket and ligand):<br>- Oxygen atoms: Red<br>- Nitrogen atoms: Blue ## Coordinate System Definition (based on front view):<br>- **X-axis**: Horizontal direction, rightward is positive<br>- **Y-axis**: Vertical direction, upward is positive<br>- **Z-axis**: Perpendicular to screen, outward is positive ## Transformation Commands:<br><br><br>1. **Rotation**: (Same as in the rotation task)<br>2. **Translation**: (Same as in the translation task)<br><br><br>## Task: The ligand in Image 7 has been displaced from its binding pose through a sequence of operations (first moved, then rotated around its own center). Analyze the spatial relationship between the pocket (Images 1-3) and the displaced ligand (Images 4-7) to determine what operations are needed to dock the ligand back into the binding pocket.<br><br>## Analysis Requirements:<br><br>- Compare the pocket orientation and geometry from Images 1-3<br>- Analyze the ligand's current position and orientation from Images 4-6<br>- Use Image 7 to understand their spatial relationship<br>- Determine the rotation needed to align the ligand's orientation with the binding pose<br>- Determine the translation needed to position the ligand into the pocket<br>- Note: Operations should be applied in order - rotation first (around ligand's center), then translation<br><br>## Output Format: [Two commands separated by comma in the exact format: roll <axis> <angle>, move <axis> <distance>] Examples:<br><br>- roll x 45, move x 17<br>- roll y -30, move x -12<br>- roll z 90, move x 9<br><br><br>Problem<br><br>[Figure 149]|
|---|

- Figure 15. Sample visualization for the ligand docking task. Zoom in for greater detail.

[Figure 150]

[Figure 151]

[Figure 152]

Top

Left Front

###### Answer: C

[Figure 153]

[Figure 154]

[Figure 155]

Right Back

Bottom

|I will provide you with six images showing the binding interaction between a protein (PDB ID: 5ey0) and a small molecule ligand from different viewing angles. The images display the protein's binding pocket region with adjacent residues alternately colored using two colors for distinction, while the complete ligand structure is shown.<br><br>## Image Description:<br><br>- **Image 1**: Front view (main perspective)<br>- **Image 2**: Left view (viewed from the left side)<br>- **Image 3**: Top view (viewed from above)<br>- **Image 4**: Back view (viewed from behind)<br>- **Image 5**: Right view (viewed from the right side)<br>- **Image 6**: Bottom view (viewed from below) ## Visualization Details:<br>- **General Rules**:<br>- Hydrogen atoms are not displayed<br>- Water molecules (H2O) are not displayed<br>- **Protein's binding pocket region**: All atoms of all constituent residues are displayed<br>- Carbon atoms can be colored using either #BF99F2 (purple) or #F2B366 (orange)<br>- The residue is numbered and labeled in the image<br>- **Ligand**: Small molecule with all atoms displayed<br>- Carbon atoms: Gray color<br>- Each atom is labeled with its atom name<br>- **Common Atom Colors** (both pocket and ligand):<br>- Oxygen atoms: Red<br>- Nitrogen atoms: Blue<br><br>## Target Hydrogen Bond: The hydrogen bond of interest is formed between:<br><br>- **Protein side**: Chain A, Residue LYS 47, Atom NZ<br>- **Ligand side**: Atom O1A ## Coordinate System Definition (based on front view):<br>- **X-axis**: Horizontal direction, rightward is positive<br>- **Y-axis**: Vertical direction, upward is positive ### Translation (Move) Rules:<br>- Command format: move <axis> <dst> (translate along the <axis> by <dst> units)<br>- **axis**: Must be x or y<br>- Positive dst: movement in positive axis direction (rightward for x-axis, upward for y-axis)<br>- Negative dst: movement in negative axis direction (leftward for x-axis, downward for y-axis)<br><br><br>## Task: Your task is to determine the translation operations needed to move the hydrogen bond to the screen center (origin point 0,0).<br><br>Specifically, you need to:<br><br>1. Locate the hydrogen bond components (chain A, residue LYS 47 and ligand atom O1A) in the images<br>2. Calculate the centroid of all atoms involved in this hydrogen bond interaction<br>3. Determine how much translation is needed to move this centroid to the screen center (0,0)<br>4. Select the correct translation command from the options below<br><br><br>## Options:<br><br>A. move x 3, move y 5 B. move x 9, move y -5 C. move x 7, move y 5 D. move x 3, move y -5<br><br>## Output Format: Please provide only the letter (A, B, C, or D) of the correct answer.<br><br>Problem<br><br>[Figure 156]|
|---|

- Figure 16. Sample visualization for the interaction location task. Zoom in for greater detail.

[Figure 157]

[Figure 158]

[Figure 159]

###### Answer

ARG 45 NH2 A, O1B; ARG 45 NE A, O2G; LYS

Top

Left Front

47 NZ A, O1A; LYS 158 NZ A, O1G; SER 43 OG A, O2B; PHE 24 N A, O6; ARG 44 N A, O3G; VAL 22 O A, N1; VAL 22 O A, N2

[Figure 160]

[Figure 161]

[Figure 162]

Right Back

Bottom

|I will provide you with six images showing the binding interaction between a protein (PDB ID: 5ey0) and a small molecule ligand from different viewing angles. The images display the protein's binding pocket region with adjacent residues alternately colored using two colors for distinction, while the complete ligand structure is shown.<br><br>## Image Description:<br><br>- **Image 1**: Front view (main perspective)<br>- **Image 2**: Left view (viewed from the left side)<br>- **Image 3**: Top view (viewed from above)<br>- **Image 4**: Back view (viewed from behind)<br>- **Image 5**: Right view (viewed from the right side)<br>- **Image 6**: Bottom view (viewed from below)<br><br>## Visualization Details:<br><br>- **General Rules**:<br>- Hydrogen atoms are not displayed<br>- Water molecules (H2O) are not displayed<br>- **Protein's binding pocket region**: All atoms of all constituent residues are displayed<br>- Carbon atoms can be colored using either #BF99F2 (purple) or #F2B366 (orange)<br>- The residue is numbered and labeled in the image<br>- **Ligand**: Small molecule with all atoms displayed<br>- Carbon atoms: Gray color<br>- Each atom is labeled with its atom name<br>- **Common Atom Colors** (both pocket and ligand):<br>- Oxygen atoms: Red<br>- Nitrogen atoms: Blue<br><br>## Task: Analyze the six images to identify all possible hydrogen bonds between the pocket and the ligand<br><br>## Output Format: List all hydrogen bonds in the following format:<br><br>**[Amino Acid Type] [Residue Number] [Donor/Acceptor Atom] [Chain ID], [Ligand Atom Name]**<br><br>### Format Rules:<br><br>- Use standard three-letter amino acid codes (e.g., ARG, GLY, SER, etc.)<br>- Always list protein residue information first, then ligand information<br>- Separate multiple hydrogen bonds with semicolons (;)<br>- Example: "ASN 460 ND2 A, O2; GLU 537 OE1 A, O2; GLU 537 OE2 A, O2; HIS 391 NE2 A, O3"<br><br>## Analysis Requirements:<br><br>- Examine all six views to get complete spatial understanding<br>- Identify atoms capable of hydrogen bonding<br>- Consider the 3D geometry from multiple perspectives<br>- Focus on interactions at typical hydrogen bonding distances<br><br><br>Please analyze the images systematically and provide the hydrogen bond list in the specified format.<br><br>Problem<br><br>[Figure 163]|
|---|

- Figure 17. Sample visualization for the pocket-ligand interaction task. Zoom in for greater detail.

