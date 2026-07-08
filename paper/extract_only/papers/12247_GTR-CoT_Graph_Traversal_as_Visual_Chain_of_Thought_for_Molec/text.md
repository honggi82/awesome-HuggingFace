# arXiv:2506.07553v3[cs.AI]13Jan2026

[Figure 1]

### GTR-VL: Graph Traversal as Visual Chain of Thought for Molecular Structure Recognition

Jingchao Wang2†, Yifan He1†, Haote Yang1†, Jiang Wu3†, Lingli Ge4, Xingjian Wei1, Yinfan Wang1, Linye Li5, Huijie Ao6, Chengjin Liu7, Bin Wang1, Lijun Wu1, Conghui He1

1Shanghai Artificial Intelligence Laboratory, 2East China Normal University, 3Peking University, 4Shanghai Jiaotong University, 5Tongji University, 6Fudan University, 7Northwestern Polytechnical University

Optical Chemical Structure Recognition (OCSR) is essential for converting molecular images into machine-readable formats. While recent vision-language models (VLMs) have shown promise, their image-captioning approach often struggles with complex molecular structures and inconsistent annotations. To address these issues, we introduce GTR-VL, featuring two key innovations: (1) the Graph Traversal as Visual Chain of Thought mechanism that emulates human reasoning by incrementally parsing molecular graphs through sequential atom-bond predictions, and (2) the data-centric Faithfully Recognize What You’ve Seen principle, which aligns abbreviated structures in images with their expanded annotations. For hand-drawn OCSR tasks, where datasets lack graph annotations and only provide final SMILES, we apply reinforcement learning using the GRPO method, introducing reward mechanisms like format reward, graph reward, and SMILES reward. This approach significantly enhances performance in hand-drawn recognition tasks through weak supervision. We developed GTR-1.3M, a large-scale instruction-tuning dataset with corrected annotations, and MolRec-Bench, the first benchmark for fine-grained evaluation of graph-parsing accuracy in OCSR. Our twostage training scheme involves SFT training for printed images and the GRPO method for transferring capabilities to hand-drawn tasks. Experiments show that GTR-VL outperforms specialist models, chemistry-domain VLMs, and commercial VLMs on both printed and handdrawn datasets.

Date: January 14, 2026 Equal contribution: Jingchao Wang, Yifan He, Haote Yang, Jiang Wu Project leader: Jiang Wu Correspondence: Conghui He, heconghui@pjlab.org.cn

### 1 Introduction

Modern chemistry’s vast knowledge is often stored in chemical molecular formulas, typically as 2D images in papers and patents, limiting machine accessibility ([35]). With the rise of Large Language Models (LLMs), converting these images for model training is crucial. Optical Chemical Structure Recognition (OCSR) technology addresses this by converting images into machine-readable formats like SMILES, crucial for digitizing chemical data and advancing AI in chemistry. Early rule-based methods ([13, 31]) were limited to simple cases, and recent deep learning advances have enhanced OCSR capabilities ([37, 12, 22, 33]). Nonetheless, further improvements are needed for handling large molecules, complex Markush structures ([23]), and hand-drawn formats.

Recently, large Vision-Language Models (VLMs) ([1, 8]) have achieved breakthroughs in visual perception ([20]), visual question answering ([52]), and multimodal reasoning ([53, 14]). They have been applied in fields like medicine ([17]), autonomous driving ([10]), remote sensing ([30, 24]), and OCR

###### SMILES

VLM or Specialist Model

O

Molecular Structure Image

Prompt

C5H11

Alkyl

OMe

(a)

###### Graph-based SMILES

Graph

Graph

###### Graph-based SMILES

###### Atoms Bonds Gen SMILES

Bond Predictor Bonds

Atoms

VLM

Atom Predictor

O

O

Molecular Structure Image

Molecular Structure Image

Prompt

C5H11

C5H11

Alkyl

Alkyl

OMe

OMe

(c)

(b)

- Figure 1: The comparison of three paradigms. (a) Image-captioning approach: Directly generates SMILES from molecular structure images using either VLMs or specialist models. (b) Graph-parsing approach: Predict atoms and bonds in separate stages to construct a molecular graph, which is then converted to SMILES. (c) Ours: Jointly generates atoms and bonds to form a molecular graph, followed by SMILES generation. The graph is then used to construct a graph-based SMILES.

([45]). Recent works such as ChemVLM ([19]), ChemDFM-X ([55]), and OCSU [11] have applied VLMs to OCSR, treating it as image captioning to generate SMILES strings. However, this approach is less effective than graph-parsing methods, and their performance on OCSR tasks needs improvement, as shown in Table 1.

After evaluating existing models, we propose two insights and design principles: (1) Graph Traversal as Visual Chain of Thought: The Chain of Thought (CoT) technique improves problem-solving by generating intermediate steps. For OCSR tasks, a visual CoT mechanism that recognizes complex molecules step-by-step can enhance performance. Unlike current methods ([22, 33, 7]) that predict atoms and bonds separately, our Graph Traversal as Visual CoT method interleaves atom and bond predictions in one pass, improving accuracy and consistency. (2) Faithfully Recognize What You’ve Seen: Abbreviations like ”Ph” for phenyl in molecular images challenge OCSR tasks. Existing methods ([33, 7]) struggle due to mismatches between images and annotations. We developed a data correction pipeline that aligns annotations with images by representing abbreviations as superatoms, improving model accuracy. Extensive experiments demonstrate that our principles significantly enhance VLM accuracy and ensure alignment with molecular structure diagrams, which improves interpretability and facilitates manual inspection and editing.

We extended these principles to hand-drawn molecular recognition tasks. Existing datasets lack fine-grained supervision for atoms and bonds, limiting model performance. We addressed this using Reinforcement Learning (RL) with the Group Relative Policy Optimization (GRPO)([41]) method, designing reward mechanisms including format, graph, and SMILES rewards.

Building on these insights, we developed GTR-1.3M, an SFT dataset for VLM OCSR tasks, containing

- 1.3 million samples with molecular images, a visual CoT process, and final SMILES strings (Figure 3). Leveraging these principles, the RL method, and the dataset, we propose a two-stage training scheme and develop GTR-VL, a specialized multimodal model. This model excels in handling complex molecular images and hand-drawn recognition tasks, advancing OCSR technology to better meet practical needs. The contributions of this paper are as follows:

- 1. We apply VLM technology to OCSR, introducing two design principles: Graph Traversal as Visual Chain of Thought and Faithfully Recognize What You’ve Seen. These principles enhance VLM accuracy and ensure alignment between molecular diagrams and images, improving interpretability and facilitating manual editing.

Input Image

Input Image

GT

GT

[Figure 2]

[Figure 3]

O=C(C([Et])[OMe])CCC(=O)O

O=C(C(Cl)[Me])CCC(=O)O

Canon. GT

Canon. GT

[Me]C(Cl)C(=O)CCC(=O)O

[Et]C([OMe])C(=O)CCC(=O)O

Consistent Inconsistent

Graph Exact Match: True SMILES Exact Match: True Graph Exact Match: True SMILES Exact Match: False

O

O

Canon. Pred.

Canon. Pred.

[Me]C(Cl)C(=O)CCC(=O)O

[OMe]C([Et])C(=O)CCC(=O)O

Et

C

C O

Cl

C

C O

C C C

C C C

Pred.

Pred.

ClC([Me])C(=O)CCC(=O)O

O=C(C([OMe])[Et])CCC(=O)O

OMe O

Me O

Pred. Graph

Pred. Graph

(a) (b)

- Figure 2: Illustration of limitations in SMILES-based evaluation. (a) The positive example: both the predicted graph and SMILES match with the ground truth. (b) The counterexample: the graph-parsing OCSR algorithm correctly interprets the molecular graph, but the SMILES does not match the ground truth and is incorrectly judged as a prediction error.

- 2. We utilize RL with the GRPO method for molecular structure recognition, introducing reward mechanisms like format, graph, and SMILES rewards. This approach effectively improves performance in recognizing hand-drawn molecular structures with only SMILES annotations.
- 3. Using these principles, we developed the VLM SFT dataset GTR-1.3M and introduced MolRecBench, a benchmark for assessing graph parsing accuracy in OCSR tasks. MolRec-Bench addresses the limitations of SMILES-based evaluations by accurately assessing structures with custom functional groups or abbreviations.
- 4. Based on the principles, RL method, and dataset, we propose the two-stage training scheme and develop the GTR-VL model, which excels with complex molecular images and hand-drawn recognition tasks.

### 2 Preliminary

#### 2.1 Image-Captioning vs Graph-Parsing

Image-captioning methods treat OCSR as an image captioning task, outputting SMILES strings directly. In contrast, graph-parsing methods predict atoms, bonds, and molecular information to construct the graph structure. Graph-parsing offers several advantages: (1) Interpretability: It allows for better algorithm optimization and robustness analysis. (2) Manual Verification: Results align with input images, enabling manual checks and semi-automated annotation. (3) Expressive Capability: It can represent complex structures like Markush structures. (4) Performance: It outperforms imagecaptioning methods with the same training data. Therefore, we chose graph-parsing for this study.

#### 2.2 Challenges of Hand-drawn OCSR

Hand-drawn molecular data poses unique challenges compared to printed depictions: the training data is much scarcer, and annotations usually only provide SMILES strings without atomic coordinates, making graph-parsing methods unsuitable. Approaches like DECIMER([38]) must rely on generating synthetic data for training, producing over 100 million synthetic samples to achieve competitive recognition within the image-captioning framework.

GTR-1.3 M

DECIMER-HD-Train

[Figure 4]

[Figure 5]

[

|HN OMe<br><br>O|
|---|

This is a schematic diagram of a chemical molecular structure. You are required to list the types of atomic elements in the diagram, the coordinates of the atoms, and the chemical bonds between all the atoms. ... Then, based on the atoms and bonds that you have listed, output a canonical SMILES in JSON format with the key 'smiles'. ...

{"a":"[OMe]","id":0,"xy":[824,133]}, #atom

- {"a":"C","id":1,"xy":[549,133]}, #atom
- {"b":1,"a1":0,"a2":1}, #bond

... ...

- {"a":"C","id":2,"xy":[412,67]}, #atom
- {"b":2,"a1":1,"a2":2}, #bond

O[C@@H] ([C@]1(OC(C(Cl)=C1)=O)C)C2=CC=CC=C2

Dataset

...

] {

... CCCCN1CCC[C@H] (OC(NC2=C(Cl)C=CC=C2C)=C)C1

"smiles": "[OMe]C1=CC(=O)NC2=CC=CC=C12"

...

}

|[Figure 6]<br><br>Graph Traversal|
|---|

|Pretrained model| |
|---|---|
| | |

Tokens

###### SFT

...

GTR-1.3 M

SFT

- (Stage1)

Reinforcement Learning

- (Stage2)

Random Selection

GTR-VL-Stage1

Reference Model

||O1|
|---|
<br><br>|O2|
|---|
<br><br>|OG|
|---|
<br><br>...| |
|---|---|
| | |

|Reward model|Format<br><br>Graph<br><br>Smiles|
|---|---|

||r1|
|---|
<br><br>|r2|
|---|
<br><br>|rG|
|---|
<br><br>...| |
|---|---|
| | |

|A1|
|---|

GTRUSPTO-4K

KL Iteration

|A2|
|---|

Group Computation

...

Policy model

DECIMERHD-Train

|AG|
|---|

- Figure 3: Our method is designed to achieve robust recognition across both printed and hand-drawn molecular structures. We begin by performing supervised fine-tuning (SFT) on the large-scale dataset GTR-1.3M to train a base model capable of recognizing printed molecular depictions. To further enhance the model’s ability to handle the challenging domain of hand-drawn inputs, we apply reinforcement learning using the GRPO algorithm on the DECIMER Hand-drawn dataset. This two-stage training pipeline enables the model to generalize effectively across different molecular representation styles.

#### 2.3 Limitations of SMILES-based Evaluation

SMILES ([46]) is a string-based language for representing molecular structures and reactions, encoding atoms and bonds in character sequences (Appendix D). OCSR evaluation datasets typically compare predicted and ground truth SMILES using exact matches or similarity measures. However, due to canonicalization issues, there are situations where the OCSR algorithm may correctly interpret the molecular graph while the SMILES might not match the ground truth and be incorrectly judged as a prediction error (Figure 2). Additionally, studies like MolScribe ([33]) and MolNexTR ([7]) replace abbreviations with an asterisk (*) in SMILES, ignoring these elements and resulting in incomplete evaluations (see Appendix B for details).

### 3 Method

#### 3.1 Insights and Design Principles

The graph-parsing OCSR task can be formulated as an image-to-graph generation task. Given an image Im containing a molecule m, we train a model f to convert Im into a molecular structure graph Gm = {a1, a2, . . . , ap, b1, b2, . . . , bq}, where ai represents the i-th atom and bj represents the j-th bond, i.e., Gm = f(Im). Compared to existing graph-parsing methods, we have two insights and corresponding design principles.

###### 3.1.1 Graph Traversal as Visual CoT

Existing graph-parsing methods ([22, 33, 7]) adopt a two-stage approach (Figure 1(b)): first predicting atoms (nodes), then chemical bonds (edges) using classifiers or GNNs. This diverges from human cognition, which naturally alternates attention between atoms and bonds. The two-stage design presents two key limitations: (1) Atom prediction lacks structural constraints from bonds, increasing ambiguity; (2) Bond prediction requires global attention over all atoms, leading to higher inference complexity and cost.

We propose graph traversal as a visual CoT, a human-inspired method that parses molecular graphs by interleaving atom and bond predictions in a single traversal (Figure 1(c)). As illustrated in Figure 3, a depth-first strategy defines the traversal order, alternating atom and bond predictions along the path. This interleaved process leverages mutual constraints: bonds are predicted based only on previously identified atoms, reducing prediction difficulty. Furthermore, the traversal serves as a visual Chain-of-Thought for VLMs, decomposing complex recognition into structured, sequential sub-tasks, thus enhancing prediction accuracy and consistency.

###### 3.1.2 Faithfully Recognizing What You’ve Seen

In molecular structure images from papers and patents, many abbreviated structures (e.g., Ph for phenyl, Pr for propyl, Bu for butyl, as shown in Figure 4) are common. While these abbreviations improve readability and conciseness, they present challenges for OCSR tasks. Existing works ([33, 7]) use annotations (from MOL files) with fully expanded molecular graphs in their patent training data. This mismatch can confuse models, leading to errors when predicting expanded forms for abbreviations seen in images (Figure 5).

We propose treating these abbreviations as ”super atoms” rather than expanding them. This faithfully recognizing what you’ve seen approach ensures consistency between images and annotations, optimizing model learning, and significantly enhancing generalization capability.

#### 3.2 GRPO for hand-drawn OCSR

As noted earlier, image-captioning methods ([36]) face challenges in hand-drawn OCSR due to the need for large-scale annotated data, while graph-parsing methods ([33]) are inapplicable because hand-drawn samples lack coordinate information. However, we observe that although coordinates are missing, topological molecular graphs can still be reconstructed from SMILES. While these coordinateindependent graphs cannot supervise token-level outputs, their structural correctness can be leveraged as a reward signal in GRPO.

Thus, we design a composite reward function integrating response format, molecular graph, and SMILES accuracy. Specifically, we compare the predicted and ground-truth graphs (from SMILES) by computing their maximum common subgraph (MCS). Graph similarity is defined as the MCS size relative to both graphs, and used as the graph-level reward, the formula is as follows where Nam, Nag, and Nap represent the number of atoms in the MCS, ground truth graph, and prediction graph respectively, and Nbm, Nbg, and Nbp denote the number of edges respectively.

Rgraph = |Nam|

+ |Nbm| |Nbg| + |Nbp|

(1)

|Nag| + |Nap|

To enforce output validity, we also include format and SMILES rewards. The final GRPO reward combines these three components as follows.

Rtotal = Rgraph + Rformat + RSMILES (2)

MolRec-Abb MolRec-USPTO Gen-SMILES Gra-SMILES Graph Gen-SMILES Gra-SMILES Graph

Model

MolScribe† 20.11 19.39 19.82 71.77 72.03 72.25 MolScribe§ 72.20 69.63 70.60 85.49 84.66 86.23

MolNexTR† 19.76 19.00 19.47 71.75 71.90 72.14 MolNexTR§ 74.60 70.98 71.85 86.54 85.30 86.96 OCSU 0.40 - - 1.71 - -

ChemVLM 4.18 - - 42.52 - -

ChemDFM-X 0.76 - - 26.94 - GPT-4o* 0.60 - - 1.60 - GPT-4o‡ 0.60 0.00 0.00 0.60 0.00 0.00

GPT-4o-mini* 0.20 - - 0.20 - GPT-4o-mini‡ 0.00 0.00 0.00 0.00 0.00 0.00

Qwen-VL-max* 0.20 - - 1.40 - Qwen-VL-max‡ 0.20 0.00 0.00 1.00 0.00 0.00

- GTR-VL-Stage1 (Ours) 84.50 84.50 85.49 91.19 91.67 93.45
- GTR-VL-Stage2 (Ours) 81.67 82.33 82.84 91.31 91.17 91.28

- Table 1: Quantitative results are reported across three exact-match metrics on two sub-benchmarks. The highest value for each metric is marked in bold. § indicates trained on GTR-1.3M; † indicates officially released checkpoint; ‡ models first predict the graph then SMILES; * models directly predict SMILES.

#### 3.3 Dataset Construction

Building on these insights, we developed GTR-1.3M, a specialized SFT dataset for VLM-based OCSR tasks. Following MolScribe and MolNexTR, GTR-1.3M is composed of two parts: (1) GTR-PubChem1M: We selected 1 million molecular SMILES from the PubChem database and used the Indigo tool to convert them into molecular images. (2) GTR-USPTO-351K: This subset was created from USPTO680K. We developed a data correction pipeline to correct and filter abbreviated structures in these samples, obtained 351k high-quality samples, and formed the GTR-USPTO-351K subset. Please refer to Appendix A for more details.

#### 3.4 Two-Stage Training

Based on the design principles, data selection, and reinforcement learning framework, we adopt a two-stage training strategy (Figure 3).

Stage 1: We perform SFT on the GTR-1.3M dataset to teach the VLM a visual CoT mechanism via molecular graph traversal. The model first constructs a molecular graph from the input image, then predicts the SMILES string. During inference, the output is parsed to recover the graph and generate SMILES via rule-based decoding. We denote SMILES directly produced by the VLM as generated SMILES, and those derived from parsed graphs as graph-based SMILES. The resulting model is referred to as GTR-VL-stage1.

Stage2: We apply GRPO using a mixture of printed and hand-drawn data: (1) GTR-USPTO-4K, sampled from GTR-USPTO-351K, and (2) DECIMER-HD-Train (4070 samples following ([26])). In Epoch 1, both the policy and reference models are initialized with GTR-VL-stage1; in subsequent epochs, they are updated using the trained model from the previous epoch. The final model is denoted as GTR-VL-stage2 or GTR-VL.

DECIMER-HD-Test ChemPix

Model

SMILES T = 1 T Mean Graph SMILES T = 1 T Mean Graph AtomLenz+EditKT∗ (AE) 28.00 33.69 48.40 29.76 39.80 48.29 60.50 48.29

DECIMER FT (v2.2) 56.48 61.79 72.90 61.10 55.46 58.08 75.00 57.75 DECIMER FT (v2.2) + AE 56.48 61.79 73.20 61.10 55.46 58.08 75.09 57.75

DECIMER HD 69.94 73.28 82.92 71.71 44.37 50.24 68.16 49.76 DECIMER HD + AE 69.94 73.28 83.00 71.71 44.37 50.24 68.36 49.76

- GTR-VL-Stage1 (Ours) 9.53 9.92 26.83 9.53 22.02 23.16 40.73 22.02
- GTR-VL-Stage2 (Ours) 75.44 80.45 86.05 75.44 86.13 86.62 91.28 86.13

- Table 2: Performance comparison on the DECIMER Hand-drawn and ChemPix datasets. AE refers to the method proposed in [26]. DECIMER+AE denotes a combined approach that integrates DECIMER with AE. SMILES indicates the proportion of samples with an exact match of the predicted and groundtruth SMILES strings. T Mean denotes the average Tanimoto similarity across all samples. T=1 indicates the proportion of samples with Tanimoto similarity = 1.

#### 3.5 Benchmark and Metrics

We developed MolRec-Bench to overcome the limitations of existing SMILES-based evaluation datasets as detailed in Section 3.2. This benchmark evaluates not only molecular graph structures but also complex scenarios like Markush structures. MolRec-Bench comprises two subsets: (1) MolRecUSPTO: Based on USPTO ([34]), it includes 5,423 molecular images from USPTO patents. (2) MolRecAbb: Derived from MolGrapher ([22]), it features 9,311 molecular images with abbreviated superatoms from USPTO 10K abb. The construction of MolRec-Bench is followed as the GTR-USPTO-351K (Section 3.3 and Appendix A). Each sample contains the original molecular image, the corrected molecular graph, and the corrected SMILES.

For MolRec-Bench, we defined three evaluation metrics: (1) Gen-SMILES: Calculates the exact match ratio by comparing canonicalized ground truth SMILES with predicted SMILES, designed for imagecaptioning-based OCSR methods. (2) Gra-SMILES: Similar to Gen SMILES but uses SMILES generated from the predicted molecular graph, suited for graph-parsing OCSR methods. (3) Graph: To compensate for the shortcomings (as mentioned in Section 2.3) of the first two measurement methods, we propose a graph-based measurement method, which measures the exact match ratio between the ground truth and predicted graphs for graph-parsing OCSR methods. This method can handle the matching of Markush structures more accurately, thereby enabling more precise evaluation. Details are shown in Appendix B.

### 4 Experiments

#### 4.1 Experiment Setup

###### 4.1.1 Baselines on printed molecule images

To evaluate our method on printed molecule images, we selected three types of baseline models: (1)Specialist Models: These are tailored for molecular structure recognition, exemplified by MolScribe ([33]) and MolNexTR ([7]). We evaluated both the open-source checkpoints and versions trained from scratch using our GTR-1.3M dataset. (2) Open-source VLMs in the Chemical Domain: These fine-tuned models, such as ChemVLM ([19]), ChemDFM-X ([55]), and OCSU ([11]), generate SMILES directly. We tested them on MolRec-Bench, reporting only the Gen-SMILES using prompts from their papers. (3) Proprietary General-purpose VLMs: We compared models like Qwen-VL-Max-2025-04-

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

###### (b)MolScribe(d)Ours(c)MolNexTR(a)Input

(b)DECIMERHD(c)Ours(a)Input

[Figure 11]

[Figure 12]

- C15H31
- C16H33

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

C13H27

[Figure 17]

[Figure 18]

C13H27

[Figure 19]

[Figure 20]

Ph Ph

C12H15

C12H15

- Figure 4: Comparison of model predictions on molecular images with abbreviated structures. Our model accurately retains abbreviations as superatoms (d) compared with (b) and (c).

Figure 5: Comparison of model predictions on handdrawn molecular images. Compared with DECIMER (b), our method (c) achieves more accurate predictions.

08 ([2]), GPT-4o-mini-2024-07-18 ([28]), and GPT-4o-2024-08-06 ([27]). Using two prompt sets, we evaluated their ability to generate SMILES directly (Gen-SMILES) and via graph parsing (Gra-SMILES). Due to interface costs, we randomly selected 500 samples from MolRec-Abb and MolRec-USPTO for evaluation.

#### 4.2 Main Results

###### 4.2.1 Training Details

Both the SFT and GRPO ([41]) training were performed on 32 NVIDIA A100 GPUs using the AdamW optimizer, with a peak learning rate of 1.6 × 10−4 and 1 × 10−5, respectively. We applied cosine decay for the learning rate and a linear warm-up for the first 10% of iterations to stabilize training. The batch size per GPU for SFT and GRPO was 2 and 4, respectively. The gradient accumulation steps for SFT and GRPO were 16 and 1, resulting in an effective batch size of 1024 and 128. All model parameters were updated during training.

###### CoT Gen SMILES

✗ 66.54 ✓ 68.85

CoT Strategy Gra SMILES Graph Atoms-then-bonds 79.02 80.15

Graph-traversal (ours) 81.88 83.26

- Table 3: The comparison of performances of whether training with visual CoT strategies or not in stage 1.

Table 4: The performance of our models under different CoT strategies. The Atoms-then-Bonds approach indicates that the model first predicts atoms followed by bonds separately during Stage 1.

Reward / Loss DECIMER-HD-Test ChemPix Format SMILESGraph SMILES T = 1 T Mean Graph SMILES T = 1 T Mean Graph

Strategy

SFT ✓ ✓ 56.09 59.04 71.68 56.19 55.14 55.79 74.14 55.14 GRPO ✓ ✓ 11.00 11.49 28.72 11.00 22.84 24.31 42.35 22.84 GRPO ✓ ✓ ✓ 75.64 79.96 86.09 75.64 82.06 83.52 88.39 82.06

Table 5: Impact of Graph Supervision on the hand-drawn OCSR task. Results demonstrate that graph supervision substantially improves transfer from printed to hand-drawn domains. While SFT-based transfer (using only SMILES) outperforms GRPO without graph supervision, both are clearly inferior to GRPO with full graph supervision.

###### 4.2.2 Overall Performance

For the printed OCSR task, we compared baseline models from Section 4.1.1 with GTR-VL, as shown in Table 1. For MolRec-USPTO and MolRec-Abb, GTR-VL-Stage1 achieved the highest performance in all 3 metrics. On MolRec-Abb, it surpassed the second-best model by 9 percentage points in all 3 metrics, highlighting its strength in handling molecular images with abbreviations. Closed-source general-purpose VLMs like GPT-4o ([27]) and Qwen ([2]), despite excelling in general vision-language tasks, performed poorly on MolRec-Bench, indicating that OCSR hasn’t been a focus in their development. Existing chemical-domain VLMs using image-captioning approaches lack the flexibility of graph-based methods and often underperform on SMILES metrics compared to earlier specialist models. This underscores the effectiveness of the approach we present. Although the performance declined after the second stage of GRPO training (which expanded the model’s recognition capabilities to hand-drawn data), it still maintained a lead over existing methods. All of our used Prompts are listed in Appendix E.

For the hand-drawn OCSR task, as shown in Table 2, GTR-VL-Stage1 (trained with SFT only) performs significantly worse than baseline models on both datasets, with SMILES reaching only 8.84% and Graph only 9.53% on DECIMER-HD-Test, indicating that the model trained only with GTR-1.3M cannot be directly applied to the task of handwritten chemical formula recognition. However, after introducing the proposed reward function and applying GRPO optimization on top of SFT, GTR-VL achieves a dramatic performance boost across all metrics. For example, on DECIMER-HD-Test, SMILES jumps from 8.84% to 75.34%, and Graph from 9.53% to 75.44%; similarly, on ChemPix, SMILES and Graph score from 22.02% to 86.13%, respectively. This fully demonstrates the effectiveness of the reward function we proposed.

###### 4.2.3 Further Analysis

As shown in Table 1, MolScribe and MolNexTR both perform well on MolRec-USPTO, achieving over 70% accuracy in Gen SMILES and Gra-SMILES. However, their performance drops below 20% on MolRec-Abb, primarily due to mismatches between abbreviated inputs and expanded ground truth graphs (see Section 3.1.2).

To ensure fairness, we retrained both models from scratch using the GTR-1.3M dataset and official code. Their performance improved notably on MolRec-Bench, validating the quality of our data and pipeline. Nevertheless, GTR-VL, powered by QwenVL ([2]), consistently outperforms them. The larger performance gap on MolRec-Abb is attributed to its greater use of abbreviations, making it more challenging.

To more intuitively demonstrate the superiority of our approach, as shown in Figure 4 and 5, we provide a visual comparison of the prediction results of MolScribe, MolNexTR, and GTR-VL-Stage1 on the MolRec-Abb dataset, as well as a visual comparison between DECIMER HD and GTR-VL-

1.0

0.9

0.8

GraphExactMatch

0.7

0.6

0.5

0.4

D | DECIMER-HD-Test

0.3

D | MolRec-Abb

0.2

D+U | DECIMER-HD-Test

0.1

D+U | MolRec-Abb

0.0

0 1 2 3 4 5 6 7

Epoch

- Figure 6: Evolution of Graph Exact Match performance over training epochs for models trained on D or D+U. D: DECIMER-HD-Train only; D+U: joint training with GTR-USPTO-4K.

Stage2 on the DECIMER-HD-Test dataset. As can be observed from the figure, our method exhibits clear advantages in both abbreviation recognition and hand-drawn molecular structure prediction.

Compared to their originally reported results, both models show significant degradation. This stems from their evaluation protocol, which treats non-expandable Markush groups as wildcard “*”, potentially inflating accuracy. Additional results and analysis are provided in Appendix C.

###### 4.2.4 Ablation Study

To further validate the effectiveness of the key design components in our proposed method, we conducted the following ablation experiments:

- (1) Effectiveness of CoT: We evaluated the impact of the CoT paradigm using the GTR-USPTO351K subset. Models were trained with either our graph traversal as visual CoT strategy or a baseline that directly predicts SMILES without CoT. As shown in Table 3, the CoT strategy yields a 2.31% improvement on Gen SMILES, indicating that decomposing the recognition task via CoT enhances both performance and stability.

- (2) Choice of CoT Strategies: We compared our graph traversal as visual CoT with an Atom-then-bonds approach, which sequentially predicts atom nodes, bond edges, and methods in MolScribe ([33]) and MolNexTR ([7]). As shown in Table 4, our approach achieves gains of 2.86% and 3.11% on Gra-SMILES, validating the effectiveness of the graph traversal strategy.
- (3) GRPO Reward Scheme: To assess our reward scheme, we compared: (a) SFT with SMILES supervision only, (b) GRPO with rewards from response format and SMILES, and (c) GRPO with rewards from response format, SMILES, and graph. As shown in Table 5, (b) struggles to generalize from GTR-1.3M to DECIMER-HD-Train due to the absence of graph-level guidance. While (a) performs reasonably well, (c) significantly outperforms both, highlighting the importance of incorporating graph-based rewards.
- (4) GRPO Training Domain: We evaluated domain impact by comparing (a) GRPO training on hand-drawn data only and (b) joint training on hand-drawn and printed data. Model (a) performs well on DECIMER-HD-Test but poorly on MolRec-Abb. In contrast, (b) improves generalization to

DECIMER-HD-Test while retaining moderate performance on MolRec-Abb. By Epoch 6, (b) matches (a)’s performance on DECIMER-HD-Test (Figure 6).

### 5 Conclusion

This paper introduces GTR-VL, a visual large language model designed for OCSR tasks, along with its accompanying SFT training dataset, GTR-1.3M. GTR-VL is developed based on two core principles: Graph Traversal as Visual Chain-of-Thought and Faithfully Recognizing What You’ve Seen. The experiments demonstrate that these innovative concepts not only significantly enhance the applicability and flexibility of OCSR models but also effectively improve model performance. Furthermore, GTR-VL robustly handles challenging scenarios, such as Markush structures. Additionally, our release of the MolRec-Bench addresses the gap in existing OCSR evaluation sets by providing a means to assess molecular graph structure parsing results. We anticipate that GTR-VL and the related dataset will drive OCSR technology toward meeting real-world needs more effectively, thereby advancing the fields of cheminformatics and AI for Science.

### References

- [1] Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A versatile vision-language model for understanding, localization, text reading, and beyond, 2023. URL https://arxiv.org/abs/2308.12966.
- [2] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025.
- [3] Henning Otto Brinkhaus, Kohulan Rajan, Achim Zielesny, and Christoph Steinbeck. Randepict: Random chemical structure depiction generator. Journal of cheminformatics, 14(1):31, 2022.
- [4] Henning Otto Brinkhaus, Achim Zielesny, Christoph Steinbeck, and Kohulan Rajan. Decimer—hand-drawn molecule images dataset. Journal of Cheminformatics, 14(1):36, 2022.
- [5] Syed Saqib Bukhari, Zaryab Iftikhar, and Andreas Dengel. Chemical structure recognition (csr) system: automatic analysis of 2d chemical structures in document images. In 2019 International Conference on Document Analysis and Recognition (ICDAR), pages 1262–1267. IEEE, 2019.
- [6] Daniel Campos and Heng Ji. Img2smi: translating molecular structure images to simplified molecular-input line-entry system. arXiv preprint arXiv:2109.04202, 2021.
- [7] Yufan Chen, Ching Ting Leung, Yong Huang, Jianwei Sun, Hao Chen, and Hanyu Gao. Molnextr: a generalized deep learning model for molecular image recognition. Journal of Cheminformatics, 16(1):141, 2024.
- [8] Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, et al. Internvl: Scaling up vision foundation models and aligning for generic visuallinguistic tasks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 24185–24198, 2024.
- [9] Djork-Arn´e Clevert, Tuan Le, Robin Winter, and Floriane Montanari. Img2mol–accurate smiles recognition from molecular graphical depictions. Chemical science, 12(42):14174–14181, 2021.
- [10] Zhizhao Duan, Hao Cheng, Duo Xu, Xi Wu, Xiangxie Zhang, Xi Ye, and Zhen Xie. Cityllava: Efficient fine-tuning for vlms in city scenario, 2024. URL https://arxiv.org/abs/2405.03194.
- [11] Siqi Fan, Yuguang Xie, Bowen Cai, Ailin Xie, Gaochao Liu, Mu Qiao, Jie Xing, and Zaiqing Nie. Ocsu: Optical chemical structure understanding for molecule-centric scientific discovery, 2025. URL https://arxiv.org/ abs/2501.15415.
- [12] Xi Fang, Jiankun Wang, Xiaochen Cai, Shangqian Chen, Shuwen Yang, Haoyi Tao, Nan Wang, Lin Yao, Linfeng Zhang, and Guolin Ke. Molparser: end-to-end visual recognition of molecule structures in the wild. arXiv preprint arXiv:2411.11098, 2024.
- [13] Igor V Filippov and Marc C Nicklaus. Optical structure recognition software to recover chemical information: Osra, an open source solution, 2009.
- [14] Junyuan Gao, Jiahe Song, Jiang Wu, Runchuan Zhu, Guanlin Shen, Shasha Wang, Xingjian Wei, Haote Yang, Songyang Zhang, Weijia Li, Bin Wang, Dahua Lin, Lijun Wu, and Conghui He. Pm4bench: A parallel multilingual multi-modal multi-task benchmark for large vision language model, 2025. URL https://arxiv.org/abs/2503.18484.
- [15] Stephen Heller, Alan McNaught, Stephen Stein, Dmitrii Tchekhovskoi, and Igor Pletnev. Inchi-the worldwide chemical structure identifier standard. Journal of cheminformatics, 5:1–9, 2013.
- [16] Ivan Khokhlov, Lev Krasnov, Maxim V Fedorov, and Sergey Sosnin. Image2smiles: Transformer-based molecular optical recognition engine. Chemistry-Methods, 2(1):e202100069, 2022.
- [17] Chunyuan Li, Cliff Wong, Sheng Zhang, Naoto Usuyama, Haotian Liu, Jianwei Yang, Tristan Naumann, Hoifung Poon, and Jianfeng Gao. Llava-med: Training a large language-and-vision assistant for biomedicine in one day, 2023. URL https://arxiv.org/abs/2306.00890.
- [18] Da-zhou Li, Xin Xu, Jia-heng Pan, Wei Gao, and Shi-rui Zhang. Image2inchi: Automated molecular optical image recognition. Journal of Chemical Information and Modeling, 64(9):3640–3649, 2024.

- [19] Junxian Li, Di Zhang, Xunzhi Wang, Zeying Hao, Jingdi Lei, Qian Tan, Cai Zhou, Wei Liu, Yaotian Yang, Xinrui Xiong, Weiyun Wang, Zhe Chen, Wenhai Wang, Wei Li, Shufei Zhang, Mao Su, Wanli Ouyang, Yuqiang Li, and Dongzhan Zhou. Chemvlm: Exploring the power of multimodal large language models in chemistry area, 2025. URL https://arxiv.org/abs/2408.07246.
- [20] Yuliang Liu, Zhang Li, Mingxin Huang, Biao Yang, Wenwen Yu, Chunyuan Li, Xu-Cheng Yin, Cheng-Lin Liu, Lianwen Jin, and Xiang Bai. Ocrbench: on the hidden mystery of ocr in large multimodal models. Science China Information Sciences, 67(12), December 2024. ISSN 1869-1919. doi: 10.1007/s11432-024-4235-6. URL http://dx.doi.org/10.1007/s11432-024-4235-6.
- [21] Joe R McDaniel and Jason R Balmuth. Kekule: Ocr-optical chemical (structure) recognition. Journal of chemical information and computer sciences, 32(4):373–378, 1992.
- [22] Lucas Morin, Martin Danelljan, Maria Isabel Agea, Ahmed Nassar, Valery Weber, Ingmar Meijer, Peter Staar, and Fisher Yu. Molgrapher: graph-based visual recognition of chemical structures. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 19552–19561, 2023.
- [23] Lucas Morin, Val´ery Weber, Ahmed Nassar, Gerhard Ingmar Meijer, Luc Van Gool, Yawei Li, and Peter Staar. Markushgrapher: Joint visual and textual recognition of markush structures. arXiv preprint arXiv:2503.16096, 2025.
- [24] Dilxat Muhtar, Zhenshi Li, Feng Gu, Xueliang Zhang, and Pengfeng Xiao. Lhrs-bot: Empowering remote sensing with vgi-enhanced large multimodal language model, 2024. URL https://arxiv.org/abs/2402. 02544.
- [25] Martijn Oldenhof, Adam Arany, Yves Moreau, and Jaak Simm. Chemgrapher: optical graph recognition of chemical compounds by deep learning. Journal of chemical information and modeling, 60(10):4506–4517, 2020.
- [26] Martijn Oldenhof, Edward De Brouwer, Adam Arany, and Yves Moreau. Atom-level optical chemical structure recognition with limited supervision. arXiv preprint arXiv:2404.01743, 2024.
- [27] OpenAI. Hello gpt-4o. https://openai.com/index/hello-gpt-4o/, 2024.
- [28] OpenAI. Gpt-4o mini: advancing cost-efficient intelligence. https://openai.com/index/ gpt-4o-mini-advancing-cost-efficient-intelligence/, 2024.
- [29] Tom Y Ouyang and Randall Davis. Chemink: a natural real-time recognition system for chemical drawings. In Proceedings of the 16th international conference on Intelligent user interfaces, pages 267–276, 2011.
- [30] Chao Pang, Xingxing Weng, Jiang Wu, Jiayu Li, Yi Liu, Jiaxing Sun, Weijia Li, Shuai Wang, Litong Feng, Gui-Song Xia, and Conghui He. Vhm: Versatile and honest vision language model for remote sensing image analysis, 2024. URL https://arxiv.org/abs/2403.20213.
- [31] Tyler Peryea, Daniel Katzel, Tongan Zhao, Noel Southall, and Dac-Trung Nguyen. Molvec: Open source library for chemical structure recognition. In Abstracts of papers of the American Chemical Society, volume 258. Amer Chemical Soc 1155 16TH ST, NW, WASHINGTON, DC 20036 USA, 2019.
- [32] Yujie Qian, Zhengkai Tu, Jiang Guo, Connor W Coley, and Regina Barzilay. Robust molecular image recognition: A graph generation approach. Technical report, Technical Report, 2022.
- [33] Yujie Qian, Jiang Guo, Zhengkai Tu, Zhening Li, Connor W Coley, and Regina Barzilay. Molscribe: robust molecular structure recognition with image-to-graph generation. Journal of chemical information and modeling, 63(7):1925–1934, 2023.
- [34] Kohulan Rajan, Henning Otto Brinkhaus, Achim Zielesny, and Christoph Steinbeck. A review of optical chemical structure recognition tools. Journal of Cheminformatics, pages 1–13, 2020. ISSN 1758-2946. doi: 10.1186/s13321-020-00465-0. URL https://doi.org/10.1186/s13321-020-00465-0.
- [35] Kohulan Rajan, Henning Otto Brinkhaus, Achim Zielesny, and Christoph Steinbeck. A review of optical chemical structure recognition tools. Journal of Cheminformatics, 12:1–13, 2020.
- [36] Kohulan Rajan, Achim Zielesny, and Christoph Steinbeck. Decimer: towards deep learning for chemical image recognition. Journal of Cheminformatics, 12(1):65, 2020.

- [37] Kohulan Rajan, Achim Zielesny, and Christoph Steinbeck. Decimer 1.0: deep learning for chemical image recognition using transformers. Journal of Cheminformatics, 13:1–16, 2021.
- [38] Kohulan Rajan, Henning Otto Brinkhaus, M Isabel Agea, Achim Zielesny, and Christoph Steinbeck. Decimer. ai: an open platform for automated optical chemical structure identification, segmentation and recognition in scientific publications. Nature communications, 14(1):5045, 2023.
- [39] Kohulan Rajan, Henning Otto Brinkhaus, Achim Zielesny, and Christoph Steinbeck. Advancements in handdrawn chemical structure recognition through an enhanced decimer architecture. Journal of Cheminformatics, 16(1):78, 2024.
- [40] Noureddin M Sadawi, Alan P Sexton, and Volker Sorge. Chemical structure recognition: a rule-based approach. In Document recognition and retrieval XIX, volume 8297, pages 101–109. SPIE, 2012.
- [41] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. Deepseekmath: Pushing the limits of mathematical reasoning in open language models, 2024. URL https://arxiv.org/abs/2402.03300.
- [42] Viktor Smolov, Fedor Zentsev, and Mikhail Rybalkin. Imago: Open-source toolkit for 2d chemical structure image recognition. In TREC, 2011.
- [43] Joshua Staker, Kyle Marshall, Robert Abel, and Carolyn McQuaw. Molecular structure extraction from documents using deep learning, 2018. URL https://arxiv.org/abs/1802.04903.
- [44] Joshua Staker, Kyle Marshall, Robert Abel, and Carolyn M McQuaw. Molecular structure extraction from documents using deep learning. Journal of chemical information and modeling, 59(3):1017–1029, 2019.
- [45] Haoran Wei, Chenglong Liu, Jinyue Chen, Jia Wang, Lingyu Kong, Yanming Xu, Zheng Ge, Liang Zhao, Jianjian Sun, Yuang Peng, Chunrui Han, and Xiangyu Zhang. General ocr theory: Towards ocr-2.0 via a unified end-to-end model, 2024. URL https://arxiv.org/abs/2409.01704.
- [46] David Weininger. Smiles, a chemical language and information system. 1. introduction to methodology and encoding rules. Journal of chemical information and computer sciences, 28(1):31–36, 1988.
- [47] Hayley Weir, Keiran Thompson, Amelia Woodward, Benjamin Choi, Augustin Braun, and Todd J Mart´ınez. Chempix: automated recognition of hand-drawn hydrocarbon structures using deep learning. Chemical science, 12(31):10622–10633, 2021.
- [48] Youjun Xu, Jinchuan Xiao, Chia-Han Chou, Jianhang Zhang, Jintao Zhu, Qiwan Hu, Hemin Li, Ningsheng Han, Bingyu Liu, Shuaipeng Zhang, et al. Molminer: you only look once for chemical structure recognition. Journal of Chemical Information and Modeling, 62(22):5321–5328, 2022.
- [49] Zhanpeng Xu, Jianhua Li, Zhaopeng Yang, Shiliang Li, and Honglin Li. Swinocsr: end-to-end optical chemical structure recognition using a swin transformer. Journal of Cheminformatics, 14(1):41, 2022.
- [50] Jiacai Yi, Chengkun Wu, Xiaochen Zhang, Xinyi Xiao, Yanlong Qiu, Wentao Zhao, Tingjun Hou, and Dongsheng Cao. Micer: a pre-trained encoder–decoder architecture for molecular image captioning. Bioinformatics, 38(19):4562–4572, 2022.
- [51] Sanghyun Yoo, Ohyun Kwon, and Hoshik Lee. Image-to-graph transformers for chemical structure recognition. In ICASSP 2022-2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 3393–3397. IEEE, 2022.
- [52] Weihao Yu, Zhengyuan Yang, Linjie Li, Jianfeng Wang, Kevin Lin, Zicheng Liu, Xinchao Wang, and Lijuan Wang. Mm-vet: Evaluating large multimodal models for integrated capabilities, 2024. URL https://arxiv. org/abs/2308.02490.
- [53] Xiang Yue, Tianyu Zheng, Yuansheng Ni, Yubo Wang, Kai Zhang, Shengbang Tong, Yuxuan Sun, Botao Yu, Ge Zhang, Huan Sun, Yu Su, Wenhu Chen, and Graham Neubig. Mmmu-pro: A more robust multi-discipline multimodal understanding benchmark, 2024. URL https://arxiv.org/abs/2409.02813.
- [54] Xiao-Chen Zhang, Jia-Cai Yi, Guo-Ping Yang, Cheng-Kun Wu, Ting-Jun Hou, and Dong-Sheng Cao. Abcnet: a divide-and-conquer based deep learning architecture for smiles recognition from molecular images. Briefings in bioinformatics, 23(2):bbac033, 2022.

- [55] Zihan Zhao, Bo Chen, Jingpiao Li, Lu Chen, Liyang Wen, Pengyu Wang, Zichen Zhu, Danyang Zhang, Yansi Li, Zhongyang Dai, Xin Chen, and Kai Yu. Chemdfm-x: towards large multimodal model for chemistry. Science China Information Sciences, 67(12), December 2024. ISSN 1869-1919. doi: 10.1007/s11432-024-4243-0. URL http://dx.doi.org/10.1007/s11432-024-4243-0.

## Appendix

### A Dataset

Building on these insights mentioned in Section 3 of the main paper, we developed GTR-1.3M, a specialized SFT dataset for VLM-based OCSR tasks. Following MolScribe and MolNexTR, GTR-1.3M is composed of two parts: (1) GTR-PubChem-1M: We selected 1 million molecular SMILES from the PubChem database and used the Indigo tool to convert them into molecular images. The difference is that we chose the offline generation method to save the generated data locally as a way to accelerate the training process. We precisely recorded the spatial positions of each atom and bond to construct the Graph Traversal process as the Chain of Thought (CoT). Following [33], we replaced specific functional groups with abbreviations to create superatoms and randomly added common R-group labels (R, R1, R2, R’, etc.) to the molecules. (2) GTR-USPTO-351K: This subset was created from USPTO-680K. We developed a data correction pipeline to correct and filter abbreviated structures in these samples, obtained 351k high-quality samples, and formed the GTR-USPTO-351K subset.

#### A.1 Graph Traversal as Visual CoT

As shown in Figure 7, the traversal starts with carbon atom 0, followed by carbon atom 1. Subsequently, the bond between atoms 0 and 1 is traversed. The process continues with carbon atom 2 and the bond between atoms 1 and 2. Following this pattern, the entire molecular graph is traversed step by step. This depth-first traversal strategy tends to prioritize branches with shallower depths.

| |
|---|

Figure 7: An example of our graph traversal order. The numbers indicate the traversal order of atoms.

#### A.2 Construction of GTR-USPTO-351K

As shown in the Figure 8, the rectification and filtering pipeline for USPTO datasets involves: (1) Image Screening: Original images from USPTO-680K are preprocessed to remove samples with multiple molecular structures, ensuring each image corresponds to a single structure to avoid ambiguity. (2) Abbreviation Extraction: OCR technology identifies and extracts chemical abbreviations (e.g., ”Ph”, ”Et”) from images for structural alignment. (3) Structure Replacement and Mapping: Using the extracted abbreviations, SMILES strings, atom sequences, and bond information in the ground truth are replaced. A mapping table between common abbreviations and their atomic structures guides this replacement. This process ensures high semantic consistency between image content and structural annotations, providing a reliable foundation for model learning. After this process, we obtained 351k high-quality samples, forming the GTR-USPTO-351K subset.

Molecular Structure Image Reconstructed Graph from GT

Mapping from Abbreviation to Atoms: [

USPTO 680K Samples

[Figure 21]

[Figure 22]

{

[Figure 23]

"abbr": "Ph", "atoms": [ "c(\d+)ccccc\1", "C(\d+)=CC=CC=C\1" ]

},{

"abbr": "C2H5", "atoms": [ "CC" ]

},{

a

"text": "(CH2)5", "atoms": [ "CCCCC" ]

Original Ground Truth

SMILES: CC(C)C[C@H](N)C(O)(C1=CC=CC=C1)C1=CC=CC=C1

},

node_coords: [[-2.6388,0.0553],[-1.9243,-0.3572],[-1.9243,-1.1822],[-1.2099,0.0553],[-0.4954,-0.3572], [-0.4954,-1.1822],[0.2191,0.0553],[0.9335,-0.3572],[0.9888,0.2615],[1.4013,-0.453], [2.2263,-0.453],[2.6388,0.2615],[2.2263,0.976],[1.4013,0.976],[0.6316,0.4678],[1.0441,-0.2467], [1.8691,-0.2467],[2.2816,0.4678],[1.8691,1.1822],[1.0441,1.1822]] bonds: [[0,1,1],[1,3,1],[3,4,1],[4,6,1],[4,5,5],[1,2,1],[6,7,1],[6,8,1],[8,9,2],[9,10,1],[10,11,2],[11,12,1], [12,13,2],[8,13,1],[6,14,1],[14,15,2],[15,16,1],[16,17,2],[17,18,1],[18,19,2],[14,19,1]]

...... ]

[Figure 24]

| | |
|---|---|
| | |

Reconstructed Graph from Replaced GT

[Figure 25]

[Figure 26]

Rule-based Abbreviation Replacement

b

[

OCR Result

[Figure 27]

{

"text": "Ph", "pos": [ 0.75, 0.01, 0.80, 0.12 ]

[Figure 28]

c

},{

"text": "Ph",

Replaced Ground Truth

SMILES: CC(C)C[C@H](N)C(O)([Ph])[Ph] node_coords: [[-2.6388, 0.0553], [-1.9243, -0.3572], [-1.9243, -1.1822], [-1.2099, 0.0553], [-0.4954, -0.3572], [-0.4954, -1.1822], [0.2191, 0.0553], [0.9335, -0.3572], [0.9888, 0.2615], [0.6316, 0.4678]] bonds: [[0, 1, 1], [1, 3, 1], [3, 4, 1], [4, 6, 1], [4, 5, 5], [1, 2, 1], [6, 7, 1], [6, 8, 1], [6, 9, 1]]"

- "pos": [ 0.88, 0.15, 0.96, 0.25 ]

},{

"text": "OH",

- "pos": [ 0.89, 0.50, 0.97, 0.60 ]

},{

"text": "NH2", "pos": [ 0.75, 0.01, 0.80, 0.125 ]

[Figure 29]

} ]

- Figure 8: Due to the discrepancies between molecular structure images and their corresponding ground truth (SMILES[46], node coords, and bonds), it is necessary to correct the ground truth in the USPTO-680K[34] dataset. The process is illustrated in Figure 8. In step a, a subset of 365k samples, each containing a single molecule, is selected from the USPTO-680K dataset. In step b, Optical Character Recognition (OCR) is applied to identify all characters in the molecular structure images. In step c, abbreviations identified by OCR that appear in the abbreviation-superatom mapping table are flagged for replacement in the ground truth. The replacement algorithm is rule-based, and its core logic determines whether the atomic combinations corresponding to each abbreviation are present in the SMILES notation, thereby guiding the decision to replace them. The input required for the replacement algorithm includes the original SMILES, node coords, bonds, the OCR results, and the abbreviation-atom mapping table, which contains common functional groups in organic chemistry. The algorithm then applies the necessary substitutions to the SMILES, node coords, and bonds to ensure consistency between the molecular structure and the ground truth. Of the 365k samples, 351k had their ground truth successfully corrected.

- A.3 Construction of GTR-PubChem-1M

Following MolScribe[33] and MolNexTR[7], we saved the images generated during their training process, because the various data enhancement operations in them may cause RDKit 1 to fail to render the correct images based on the generated SMILES. Therefore, although our total number of seed SMILES is 1,000,000, the final total number of samples obtained is 999,950. Details of the data enhancement and generation process can be found in MolScribe.

- A.4 Comparison of GTR-USPTO-351K and USPTO-680K

- Figure 9 compares the ground truth of GTR-USPTO-351K and USPTO-680K. The strict alignment between molecular structure images and reconstructed graphs entails precise prediction results.

1http://www.rdkit.org/

###### Image USPTO-680K GTR-CoT-USPTO-351K SMILES

[Figure 30]

[Figure 31]

[Figure 32]

USPTO-680K: O=C1C(C2=CC=CC=C2)=C(C2=CC= CC=C2)N2C(C3=CC=CC=C3)=C(C3

=CC=CC=C3)C(=O)N12 GTR-CoT-USPTO-351K: O=C1C([Ph])=C([Ph])N2C([Ph])=C([P h])C(=O)N12

[Figure 33]

USPTO-680K: [C-]#[N+]C1=CC=C(C2=CC=C([ N+]#[C-])C=C2)C=C1

[Figure 34]

[Figure 35]

GTR-CoT-USPTO-351K: [NC]C1=CC=C(C2=CC=C([CN])C=C 2)C=C1

USPTO-680K: CCCCCCCCCCCCCCCCCCCCOc1c cc(/N=C(\C)c2cccc(/C(C)=N/c3ccc(O CCCCCCCCCCCCCCCCCCCC)cc3C) n2)c(C)c1

[Figure 36]

[Figure 37]

[Figure 38]

GTR-CoT-USPTO-351K: [C20H41]Oc1ccc(/N=C(\C)c2cccc(/C (C)=N/c3ccc(O[C20H41])cc3C)n2)c(C) c1

[Figure 39]

[Figure 40]

[Figure 41]

USPTO-680K: CC(C)C1N(C)C(=O)OC1(C1=CC=CC= C1)C1=CC=CC=C1

GTR-CoT-USPTO-351K: CC(C)C1N(C)C(=O)OC1([Ph])[Ph]

- Figure 9: The visualization result of USPTO-680K and GTR-USPTO-351K. The reconstructed graphs of USPTO-351K (the second column) are unaligned with the molecular structure images. GTR-USPOTO351K, however, strictly follow the molecular structure in the images. We use different colors to mark SINGLE, DOUBLE, TRIPLE, and AROMATIC bonds, as well as different colored arrows for BEGINWEDGE and BEGINDASH bonds.

B Evaluation

- B.1 Evaluation details of OCSR methods

Figure 10 demonstrates the details of the evaluation of Molscribe and MolNexTR. The abbreviations of the functional groups occurring in the molecular structure images are simply replaced by * during the evaluation, resulting in the degradation of the evaluation accuracy.

- B.2 Graph-based Metric

- Figure 11 demonstrates the details of the evaluation of our graph-based metric. The abbreviations of the functional groups are kept as it is. Meanwhile, the predicted and ground truth graphs are compared directly, instead of comparing the SMILES generated by these graphs, leading to a more accurate and direct evaluation diagram.

###### MolScribe & MolNexTR Judge Pipeline

Pred SMILES Canonical Pred1: [R5]C([R2])C(=O)O

Replace special atoms [R5] → [5*] [R2] → [2*] Pred1: [2*]C([5*])C(=O)O [MeO] → [*] [Me] → [*] Pred2: *C(*)C(=O)O

Pred2: [MeO]C([Me])C(=O)O

SMILES Matching

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

Pred1 GT1 Pred2 GT2

GT SMILES

Replace special atoms [R5] → [5*]

Canonical

[Figure 46]

[R2] → [2*] GT1: [2*]C([5*])C(=O)O

- GT1: [R5]C([R2])C(=O)O

- GT2: [MeO]C([Et])C(=O)O

[MeO] → [*] [Et] → [*]

GT2: *C(*)C(=O)O

[Figure 47]

[Figure 48]

[Figure 49]

- Figure 10: The evaluation process of MolScribe and MolNexTR. All the abbreviations of functional groups are replaced by * when comparing the predicted and ground truth SMILES.

### C More Experiment Results

- C.1 More Prediction Visualization

Figure 12 shows more prediction visualization results of MolScribe, MolNexTR, and our method.

- C.2 Bad Case Analysis

- Figure 13 shows some failure cases of our method on MolRec-Abb. In case 1, there is a problem of incorrectly predicting the order of the abbreviations. In case 2. The left side predicted ”HOOC” retains the abbreviated form, but on the right side, ”COOH” does the expansion form. In addition. In case 3, the model misses the topmost ”Ac”. This is mainly because there are too few samples of such a writing style, and the model has difficulty in predicting the abbreviations above and below most of the time. In case 4, the superatom ”Cbz” on the far right is repeatedly positioned on a carbon atom.
- Figure 14 shows some failure cases of our method in DECIMER-HD-Test. In Case 1 and Case 2, our three-dimensional model failed to correctly predict the chiral hydrogen bond, which may be due to the small number of triple bond samples in the training dataset. In Case 2 and Case 3, the model failed to predict the chiral hydrogen bond, which is the main direction of our future research.

### D Chemoinformatics Basics

#### D.1 SMILES

SMILES is a method for representing the structure of a molecule as an ASCII string. David Weininger initially proposed the concept in the 1980s to represent and store information about chemical molecules in computers. The fundamental principle of SMILES is to describe the molecular topology (i.e., the manner in which atoms are interconnected) in a single line of string. This representation has applications in areas such as databases, machine learning, molecular searching, and cheminformatics.

Common organic atoms can be used directly with their atomic symbols (e.g. ”C”, ”N”, ”O”, ”S”, ”P”, ”F”, ”Cl”, ”Br”, ”I”). Special or charged atoms are denoted using square brackets, for example, ”[Na+]”, ”[Fe+3]”, and ”[C@H]”. In SMILES, single bonds are typically not represented, double bonds are indicated by ”=”, triple bonds by ”#”, and aromatic bonds by ”:”. Such as ethane: ”CC”, ethylene: ”C=C”, and acetylene: ”C#C”. In SMILES, parentheses are frequently employed to denote

###### Our Judge Pipeline

Pred. Graph

Construct graph

- Pred 1: R5

R2

C C

O

O

- Pred 2: MeO

- Pred 1: Atoms, Bounds

- Pred 2: Atoms, Bounds

Me

O

Graph Matching

[Figure 50]

C C

O

[Figure 51]

[Figure 52]

Pred1 GT1 Pred2 GT2

GT SMILES

Convert to graph

[Figure 53]

- GT 1: R5

R2

C C

O

O

- GT 2: MeO

- GT 1: [R5]C([R2])C(=O)O

- GT 2: [MeO]C([Et])C(=O)O

Et

O

[Figure 54]

C C

O

[Figure 55]

- Figure 11: The evaluation process of our Graph-based metric. All the functional groups and Markush structures are kept in their abbreviations when comparing the predicted and ground truth graphs.

branched chains, such as isopropanol: ”CC(C)O”. Numbers are employed in SMILES to denote the commencement and cessation of atoms, thus forming a ring. For example: cyclohexane: ”C1CCCC1”, benzene: ”c1ccccc1”, (the lowercase ”c” represents aromatic carbon). SMILES also supports the description of chiral centres; for example, ”[C@H]” denotes a clockwise configuration, while ”[C@@H]” denotes a counterclockwise configuration.

In comparison with structural formulae or molecular diagrams, SMILES representations are characterised by their conciseness and ease of use for database storage, searching, and chemical calculations. Furthermore, the majority of SMILES can be reduced to structural formulas and are supported by numerous chemical software applications (e.g., RDKit, Open Babel2).

#### D.2 SMILES Canonicalization

SMILES canonicalization is defined as the process of converting a molecular structure into a unique and canonical SMILES representation. ”CCO” and ”OCC” are correct for ethanol, but they are frequently required to possess a single, distinct SMILES for a given molecule within databases or algorithms. In this instance, the utilisation of canonical SMILES is imperative to establish a benchmark. The implementation of canonical SMILES is essential for the standardisation of SMILES. This approach eliminates the need for duplicated data, expedites the process of retrieving and comparing information, and facilitates the learning and chemical modelling processes. Implementing the canonicalization of SMILES is determined by the algorithm, not by the habits of human-written SMILES. It consists of 3 main steps:

- 1. Molecular graph renumbering: View molecules as graph structures (atoms are nodes and bonds are edges) and renumber atoms, giving each atom an index.
- 2. Topology-based sorting: sort the molecules according to their structural rules and chemical properties (e.g. atomic number, connectivity, ring structure, etc.) and choose a minimum or optimal order.
- 3. Follow the path after sorting to generate a SMILES string in standard form. 2http://openbabel.org/

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

##### (a) Input (b) MolScribe (c) MolNexTR (d) Ours

- Figure 12: More prediction visualization of MolScribe, MolNexTR, and our method. We use different colors to mark SINGLE bonds, DOUBLE bonds, TRIPLE bonds, and AROMATIC bonds, as well as different colored arrows for BEGINWEDGE bonds and BEGINDASH bonds.

However, due to the uncertainty of the chemical nature of the abbreviated structures and the fact that they cannot be exhaustively enumerated to give a fixed index, it is not possible to distinguish between the two abbreviated structures when they are both located at the endpoints of the molecule. This results in an Incorrect judgment when using canonical SMILE for Exact Match.

### E All prompts

#### E.1 GTR’s Prompts

The following three figures demonstrate the prompt used by our model. Figure 15 demonstrates the prompt for the direct prediction. Figure 16 demonstrates the prompt for predicting the atoms first, then bonds, and finally SMILES. Figure 17 demonstrates the prompt for predicting the atoms and bonds in a graph traversal manner first, then predicting SMILES.

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

1 2 3 4

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

Input Prediction Input Prediction

- Figure 13: Some typical bad cases of our method on MolRec-Abb. We use different colors to mark SINGLE bonds, DOUBLE bonds, TRIPLE bonds, and AROMATIC bonds, as well as different colored arrows for BEGINWEDGE bonds and BEGINDASH bonds.

#### E.2 Proprietary VLMs’ Prompts

The following two figures demonstrate the prompt used by three proprietary VLMs (Qwen-VL-Max2025-04-08[2], GPT-4o-mini-2024-07-18[28], and GPT-4o-2024-08-06[27]). Figure 18 demonstrates the prompt for the direct prediction of SMILES. Figure 19 demonstrates the prompt for predicting the atoms and bonds in a graph traversal manner first, then predicting SMILES.

### F Related Works

#### F.1 OCSR Methods

OCSR methods are generally categorized into image-captioning and graph-parsing approaches. Imagecaptioning methods treat OCSR as an image captioning task, directly outputting SMILES strings. These models use an encoder to extract visual features and a decoder to generate SMILES[46] or InChI[15] sequences. Early models combined convolutional encoders with recurrent decoders (RNN, GRU, or LSTM), such as MSE-DUDL[44], DECIMER[36], Img2Mol[9], ChemPix[47], and MICER[50]. Later works introduced transformer-based architectures, including DECIMER 1.0[37], DECIMER 2.0[38], SwinOCSR[49], IMG2SMI[6], Image2SMILES[16], Image2InChI[18], and MolParser[12]. Recently, vision-language models (VLM) have been applied to this task, as seen in ChemVLM[19], ChemDFMX[55], and OCSU[11].

Graph-parsing methods predict atoms, bonds, and additional information (e.g., charges) from images to derive molecular graph structures. Early methods used hand-crafted rules for component detection and graph reconstruction[5, 13, 21, 29, 31, 40, 42]. While effective in simple, noise-free scenarios, these methods struggle with complexity and have high maintenance costs. Recent methods leverage deep learning for component detection or segmentation[25, 48, 54], and more recent approaches use deep learning to construct graphs directly[32, 51]. Existing graph-parsing methods, such as MolGrapher[22], MolScribe[33], and MolNexTr[7], use a two-stage approach (Figure 1 of the main paper). They first predict atoms (nodes) and then chemical bonds (edges) using classifiers or graph neural networks. This approach results in redundant computations and increased task complexity.

|1<br><br>[Figure 80]<br><br>[Figure 81]<br><br>|[Figure 82]<br><br>[Figure 83]<br><br>2|
|---|---|
|[Figure 84]<br><br>[Figure 85]<br><br>3|[Figure 86]<br><br>[Figure 87]<br><br>4|

###### Input Prediction Input Prediction

- Figure 14: Some typical bad cases of our method on DECIMER-HD-Test. We use different colors to mark SINGLE bonds, DOUBLE bonds, TRIPLE bonds, and AROMATIC bonds, as well as different colored arrows for BEGINWEDGE bonds and BEGINDASH bonds.

<image> You are viewing a diagram of a chemical molecular structure. You should output a canonical SMILES string of the molecule in JSON format with the key 'smiles'. Example format:

```json { "smiles": "C1=CC=CC=C1"}``` Strictly follow the given format and do not add any extra explanations or content."}]

Figure 15: Prompt for our model to directly predict SMILES.

#### F.2 OCSR Datasets and Evaluation Benchmarks

OCSR datasets and benchmarks are categorized into synthetic and real images. In synthetic datasets, tools like RDKit or Indigo generate molecular images from chemical database SMILES [33, 7, 12]. In [39], RanDepict[3] is used to synthesize images that simulate handwritten forms, enhancing model performance on actual handwritten test sets.

Real image datasets contain images from patents and literatures. For example, real data from USPTO patent documents are adopted in [33, 7]. [4] is a real handwritten dataset containing 5088 samples. Many evaluation datasets come from real patents or papers. The review article[35] organizes multiple evaluation sets from real patents or papers. [43, 33, 22] also introduce evaluation sets from USPTO patents or journals. Most training sets and all evaluation sets focus on image-captioning methods, containing only images and corresponding SMILES. Thus, even graph-parsing methods rely on comparing predicted SMILES with ground truth, limiting direct evaluation.

### G Limitation

While GTR-VLM achieves strong results on MolRec-Abb and MolRec-USPTO, further improvements are possible through the integration of advanced reinforcement learning strategies and the development of higher-quality datasets. Despite these opportunities for enhancement, our approach constitutes a pioneering contribution to the field, offering a novel, robust solution for OCSR and providing meaningful guidance for downstream applications such as automated chemical literature analysis.

<image> You are viewing a diagram of a chemical molecular structure. First, list all the atom types and their coordinates from the image, followed by detailing all the chemical bonds. The types of chemical bonds include ['single', 'double', 'triple', 'aromatic', 'solid wedge', 'dashed wedge']. For wedge bonds, the direction is drawn from atom1 to atom2: a solid wedge indicates that atom2 is protruding out of the plane towards the observer; while a dashed wedge indicates that atom2 is receding into the plane away from the observer. Present the results in JSON list format without any additional text. Example format: ```json { "atoms": [ {"id": 0, "atom": "C", "point_2d": [x1, y1] }, {"id": 1, "atom": "H", "point_2d": [x2, y2]},

... ], "bonds": [ { "atom1": 0, "atom2": 1, "bond_type": "single" },

- { "atom1": 0, "atom2": 1, "bond_type": "single" },
- { "atom1": 0, "atom2": 2, "bond_type": "double" },

- { "atom1": 1, "atom2": 3, "bond_type": "solid wedge"},
- { "atom1": 2, "atom2": 4, "bond_type": "dashed wedge"},

... ] } ``` Strictly follow the given format and do not add any extra explanations or content." Finally, based on the atoms and bonds that you have listed, you should output a canonical SMILES string of the

molecule in JSON format with the key 'smiles'. Example format: ```json { "smiles": "C1=CC=CC=C1" } ``` Again, strictly follow the given format and do not add any extra explanations or content."

Figure 16: Prompt for our model to predict the atom first, then bonds, and finally SMILES.

<image> This is a schematic diagram of a chemical molecular structure. You are required to list the types of atomic elements in the diagram, the coordinates of the atoms, and the

chemical bonds between all the atoms. In your predicted results, the type of bond is replaced by a number, with a single bond being 1, a double bond being 2, a triple bond being 3, an aromatic bond being 4, a solid wedge being 5, and a dashed wedge being 6. displays the results in JSON list format strictly following the format below. ```json [

- {"a": "C", "id": 0, "xy": [x1, y1]},
- {"b": 1, "a1": 0, "a2": 1,},

] ``` a means atom, xy means coordinates, b means bond, a1 and a2 means the atom id of the bond. Then, based on the atoms and bonds that you have listed, output a canonical SMILES in JSON format with the

key 'smiles'. Example format: ```json {

"smiles": "C1=CC=CC=C1"

} ``` Again, strictly follow the given format and do not add any extra explanations or content."

Figure 17: Prompt for our model to predict atoms and bonds in a graph traversal manner first, then predict SMILES (Our method).

###### User:

You are viewing a diagram of a chemical molecular structure. Your task is to generate a SMILES string of the molecule. Present the results in JSON format like the case given below. Note that you need to predict strictly what is in the image, and if there are multiple atoms written together, treat it as a superatom and wrap it in SMILES using middle brackets.

###### Assistant:

OK, please provide your image. User:

<shot image> Assistant:

```json {"smiles": "[MeO]C1=CN=C(C)C=C1"}``` User:

<test image>

Figure 18: Prompt for proprietary LLMs to directly predict SMILES.

###### User:

You are viewing a diagram of a chemical molecular structure. First, list all the atom types and their coordinates from the image, followed by detailing all the chemical bonds. The types of chemical bonds include ['single', 'double', 'triple', 'aromatic', 'solid wedge', 'dashed wedge']. For wedge bonds, the direction is drawn from atom1 to atom2: a solid wedge indicates that atom2 is protruding out of the plane towards the observer; while a dashed wedge indicates that atom2 is receding into the plane away from the observer. Present the results in JSON list format without any additional text. Example format: ```json{"atoms": [ {"id": 0, "atom": "C", "point_2d": [x1, y1]}, {"id": 1, "atom": “H", "point_2d": [x2, y2]}, ...], "bonds": [ {"atom1": 0,"atom2": 1,"bond_type": "single"}, {"atom1": 1,"atom2": 2,"bond_type": "single"}, {"atom1": 0,"atom2": 2,"bond_type": "double"}, ...]}``` \n Strictly follow the given format and do not add any extra explanations or content." Finally, based on the atoms and bonds that you have listed, you should output a canonical SMILES string of the molecule in JSON format with the key 'smiles'. Example format:```json{"smiles": "C1=CC=CC=C1"}``` Again, strictly follow the given format and do not add any extra explanations or content. Note that you need to predict strictly what is in the image, and if there are multiple atoms written together, treat it as a superatom and wrap it in SMILES using middle brackets. Assistant:

OK, please provide your image. User:

<shot image> Assistant:

```json{ "atoms": [ {"id": 0, "atom": "[R3]", "point_2d": [592, 1]}, {"id": 1, "atom": "S", "point_2d": [474, 35]}, {"id": 2, "atom": "C", "point_2d": [474, 104]}, {"id": 3, "atom": "C", "point_2d": [355, 138]}, {"id": 4, "atom": "C", "point_2d": [355, 207]}, {"id": 5, "atom": "N", "point_2d": [237, 241]}, {"id": 6, "atom": "[R4]", "point_2d": [118, 207]}, {"id": 7, "atom": "C", "point_2d": [321, 271]}, {"id": 8, "atom": "O", "point_2d": [184, 271]}, {"id": 9, "atom": "O", "point_2d": [389, 330]}, {"id": 10, "atom": "C", "point_2d": [526, 330]}, {"id": 11, "atom": "C", "point_2d": [594, 271]}, {"id": 12, "atom": "C", "point_2d": [355, 207]}, {"id": 13, "atom": "O", "point_2d": [355, 138]}, {"id": 14, "atom": "O", "point_2d": [236, 241]}, {"id": 15, "atom": "C", "point_2d": [118, 207]}, {"id": 16, "atom": "C", "point_2d": [0, 241]} ], "bonds": [ {"atom1": 6, "atom2": 5, "bond_type": "single"}, {"atom1": 5, "atom2": 4, "bond_type": "single"}, {"atom1": 4, "atom2": 3, "bond_type": "single"}, {"atom1": 3, "atom2": 2, "bond_type": "single"}, {"atom1": 2, "atom2": 1, "bond_type": "single"}, {"atom1": 1, "atom2": 0, "bond_type": "single"}, {"atom1": 4, "atom2": 7, "bond_type": "single"}, {"atom1": 7, "atom2": 8, "bond_type": "double"}, {"atom1": 7, "atom2": 9, "bond_type": "single"}, {"atom1": 9, "atom2": 10, "bond_type": "single"}, {"atom1": 11, "atom2": 10, "bond_type": "single"}, {"atom1": 4, "atom2": 12, "bond_type": "single"}, {"atom1": 12, "atom2": 13, "bond_type": "double"}, {"atom1": 12, "atom2": 14, "bond_type": "single"}, {"atom1": 14, "atom2": 15, "bond_type": "single"}, {"atom1": 16, "atom2": 15, "bond_type": "single"} ]}``` ```json{"smiles": "[R3]SCCC(N[R4])(C(=O)OCC)C(=O)OCC"}``` User:

<test image>

###### Figure 19: Prompt for proprietary LLMs to predict the atoms and bonds in a graph traversal manner first, then predicting SMILES.

