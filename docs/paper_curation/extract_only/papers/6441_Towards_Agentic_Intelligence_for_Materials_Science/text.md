Towards Agentic Intelligence for Materials Science 1 Huan Zhang1,2, Yizhan Li1,2, Wenhao Huang1,2, Ziyu Hou4, Yu Song1,2, Xuye Liu4, Farshid 2 Effaty1,2,5, Jinya Jiang8, Sifan Wu1,2, Qianggang Ding1,2, Izumi Takahara6, Leonard R. 3 MacGillivray1,5, Teruyasu Mizoguchi6, Tianshu Yu7, Lizi Liao12, Yuyu Luo9, Yu Rong10, Jia Li9, 4 Ying Diao3, Heng Ji3, and Bang Liu1,2,11,* 5

# arXiv:2602.00169v2[cond-mat.mtrl-sci]6Feb2026

1DIRO & Institut Courtois, Universite´ de Montreal´ 6 2Mila – Quebec AI Institute 7

- 3University of Illinois Urbana-Champaign 8
- 4University of Waterloo 9
- 5Universite´ de Sherbrooke 10
- 6The University of Tokyo, Institute of Industrial Science 11
- 7The Chinese University of Hong Kong, Shenzhen 12 8University of California, San Diego 13 9The Hong Kong University of Science and Technology, Guangzhou 14 10Alibaba DAMO Academy 15 11Canada CIFAR AI Chair 16 12Singapore Management University 17

*Correspondence: bang.liu@umontreal.ca 18 Abstract 19

The convergence of artificial intelligence and materials science presents a transforma- 20 tive opportunity, but achieving true acceleration in discovery requires moving beyond task- 21 isolated, fine-tuned models toward agentic systems that plan, act, and learn across the full 22 discovery loop. This survey advances a unique pipeline-centric view that spans from cor- 23 pus curation and pre-training, through domain adaptation and instruction tuning, to goal- 24 conditioned agents interfacing with simulation and experimental platforms. Unlike prior re- 25 views, we treat the entire process as an end-to-end system to be optimized for tangible dis- 26 covery outcomes rather than proxy benchmarks. This perspective allows us to trace how 27 upstream design choices—such as data curation and training objectives—can be aligned 28 with downstream experimental success through effective credit assignment. 29

To bridge communities and establish a shared frame of reference, we first present an inte- 30 grated lens that aligns terminology, evaluation, and workflow stages across AI and materials 31 science. We then analyze the field through two focused lenses: From the AI perspective, 32 the survey details Large Language Model(LLM) strengths in pattern recognition, predictive 33 analytics, and natural language processing for literature mining, materials characterization, 34 and property prediction; from the materials science perspective, it highlights applications in 35 materials design, process optimization, and the acceleration of computational workflows via 36 integration with external tools (e.g., density functional theory (DFT), robotic labs). Finally, we 37 contrast passive, reactive approaches with agentic design, cataloging current contributions 38 while motivating systems that pursue long-horizon goals with autonomy, memory, and tool 39 use. This survey charts a practical roadmap towards autonomous, safety-aware LLM agents 40 aimed at discovering novel and useful materials. 41

## Contents 42

- 1 Introduction 3 43
- 2 Recent Progress of AI and LLMs 5 44

- 2.1 Deep Learning Revolution . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 6 45
- 2.2 Transformer and Foundation Models . . . . . . . . . . . . . . . . . . . . . . . . . . 8 46
- 2.3 LLMs and Agents . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 10 47
- 2.4 Pipeline-Centric Perspective . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 12 48

- 3 Reactive Tasks in Materials Science from an AI Perspective 14 49

- 3.1 Prediction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 14 50

- 3.1.1 Regression Tasks . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 14 51
- 3.1.2 Classification Tasks . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 18 52
- 3.1.3 Advanced Methodologies . . . . . . . . . . . . . . . . . . . . . . . . . . . . 19 53

- 3.2 Mining . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 20 54

- 3.2.1 Information Extraction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 20 55
- 3.2.2 Knowledge Graph . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 21 56
- 3.2.3 Database Automation and Annotation . . . . . . . . . . . . . . . . . . . . . 21 57

- 3.3 Generation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 22 58

- 3.3.1 Structure Generation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 22 59
- 3.3.2 Inverse Design . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 23 60
- 3.3.3 Synthesis Route Generation . . . . . . . . . . . . . . . . . . . . . . . . . . 23 61

- 3.4 Optimization and Verification . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 24 62

- 3.4.1 Materials Discovery Process Optimization . . . . . . . . . . . . . . . . . . . 24 63
- 3.4.2 Simulation and AI-based Verification . . . . . . . . . . . . . . . . . . . . . . 24 64
- 3.4.3 Agent-based Closed-loop Labs . . . . . . . . . . . . . . . . . . . . . . . . . 24 65

- 3.5 Data and Knowledge . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 25 66

- 3.5.1 Data scarcity and heterogeneity . . . . . . . . . . . . . . . . . . . . . . . . 26 67
- 3.5.2 Knowledge integration . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 27 68
- 3.5.3 Multimodality . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 29 69

- 3.6 Explainability . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 31 70

- 3.6.1 Sparse and Closed-form Models . . . . . . . . . . . . . . . . . . . . . . . . 32 71
- 3.6.2 Attention and Graph Explainers . . . . . . . . . . . . . . . . . . . . . . . . . 32 72
- 3.6.3 Physics-informed Interpretability . . . . . . . . . . . . . . . . . . . . . . . . 33 73

- 3.7 Pipeline-Centric Perspective . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 33 74

- 4 Agentic Systems for Materials Science 34 75

- 4.1 Contemporary Agentic Systems in Materials Discovery . . . . . . . . . . . . . . . . 36 76
- 4.2 Scientist AI: Beyond Data Fitting to Scientific Reasoning . . . . . . . . . . . . . . . 37 77

- 4.2.1 Hypothesis Generation and Search Over Scientific Spaces . . . . . . . . . 38 78
- 4.2.2 Scientific Critical Thinking . . . . . . . . . . . . . . . . . . . . . . . . . . . . 39 79
- 4.2.3 Experiment and Simulation Planning (Decision-Making Under Uncertainty) 39 80
- 4.2.4 Interpretation, Explanation, and Hypothesis Revision . . . . . . . . . . . . . 40 81

- 4.3 Human–AI Collaboration in Scientific Workflows . . . . . . . . . . . . . . . . . . . . 41 82
- 4.4 Pipeline-Centric Perspective . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 43 83

- 5 Discussion & Conclusion 44 84

## 1 Introduction 85

Materials science faces the primary challenges of integrating insights from an expanding body 86 of literature and ultimately expediting the autonomous discovery of novel functional materials. 87 To solve these problems, we need to combine knowledge from different types of data, different 88 scales, and different experimental settings. Conventional uses of machine learning in materi- 89 als science, such as the initial use of LLMs, have mostly followed a static model: models are 90 trained on a set of curated datasets to do specific tasks, like predicting properties or extracting 91 entities1–3. Pre-trained LLMs excel at text understanding, efficiently mining chemical notation, ex- 92 perimental protocols, and technical jargon from unstructured text, thereby improving knowledge 93 extraction and hypothesis generation from the literature4,5. Emerging multimodal frameworks 94 that integrate textual, graph-based, and image data enrich molecular and materials representa- 95 tions for retrosynthesis planning, catalyst design, and structure–property understanding across 96 scales6–8. Instruction-tuning or other post-training approaches further bridge the gap between 97 human intuition and machine execution by following complex, user-defined workflows in mate- 98 rials research; yet, they remain constrained when operating without closed-loop feedback and 99 long-horizon objectives. Consequently, progress toward autonomous discovery remains limited 100 not by a lack of powerful models, but by the absence of an integrated framework that connects 101 these isolated capabilities to improved experimental outcomes. 102

This disconnection between static training proxies and dynamic utility reflects a broader 103 pathology in AI evaluation. According to METR’s RE-Bench analysis9, frontier models resorted 104 to sophisticated exploits, for example, monkey-patching the time function to spoof evaluation. 105 Echoing the “reward is enough” hypothesis10 in reinforcement learning (RL), which argues that 106 maximizing a suitably defined scalar reward can, in principle, drive the emergence of general 107 intelligence, we posit that a carefully designed, end-to-end discovery reward, i.e., grounded in 108 the successful identification and validation of novel, useful materials, is in principle sufficient to 109 shape the entire AI4MatSci pipeline toward our ultimate scientific objective. It is not merely high 110 benchmark performance but the autonomous generation of novel and useful materials through 111 a result-oriented, end-to-end system capable of robust inference, adaptive experimentation, and 112 safe deployment. In classical deep learning, end-to-end refers to the fact that a single scalar 113 loss defined at the final task output is back-propagated through all differentiable components of 114 the model and feature stacks, enabling joint optimization of representation learning, intermedi- 115 ate modules, and prediction heads without manually engineered interfaces. Unfortunately, the 116 traditional reactive and phase-centric approach confines the model’s capability to operate to indi- 117 vidual stages of the entire workflow, making it difficult to adapt across dynamic, multi-modal, and 118 potentially lifelong workflows that are essential for achieving autonomous materials discovery. 119 Furthermore, today’s LLMs lack long-horizon reasoning and seamless integration of multi-scale, 120 multimodal data streams, which are indispensable for driving an end-to-end materials discov- 121 ery pipeline in realistic settings. Our central thesis extends this notion of end-to-end beyond a 122 single neural network to the full materials discovery pipeline in Fig. 1, treating pre-training, do- 123 main adaptation, tool use, experiment planning, and autonomous labs not as fixed stages but as 124 coupled, trainable components linked by credit assignment propagated directly from real-world 125 discovery outcomes. 126

Through the pipeline-centric lens, as shown in Fig. 1, this survey advocates shifting AI4MS 127 from static fine-tuning to goal-oriented agent workflows for materials discovery. The discussion 128 is structured around three guiding questions: Q1 (Evolution of Agency): How have general- 129 purpose machine learning models evolved from merely passive processors to agentic systems? 130 Q2 (Limitations of Phase-Centric Approaches) Why are existing AI for materials science sys- 131 tems not enough to meet the needs of autonomous materials discovery, even though they score 132

Section 3

Section 4

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

Signal back-propagation from different sources

Explainability

Agentic System

Scientist AI

Human-AI

Transition from different stages (e.g., pre-training to ﬁne-tuning)

[Figure 5]

EraofOpen-EndedExperimentalDataEraofHumanGeneratedData

[Figure 6]

Open-ended Experiments

Action Operator applying actions to the operand

Mining

Monitor

[Figure 7]

RL

[Figure 8]

Human

Agentic LLM

Section 2

Tools

Improve

Optimization

[Figure 9]

[Figure 10]

Call

[Figure 11]

[Figure 12]

LLM Agents

Data & Knowledge

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

Foundation Models

Prediction

Pre-Trained LLM

MatSci-Tuned LLM

[Figure 17]

[Figure 18]

Pre-Train Fine-Tune

General Tasks

Deep Learning

Domain Tasks

[Figure 19]

[Figure 20]

Generation

- Figure 1: An overview of the key sections and an illustrative end-to-end pipeline encompasses key elements like general pre-training tasks & data, foundation language models, domainspecific tasks & data, materials-oriented model adaptation, goal-driven Large Language Model (LLM) agents, and iterative open-ended materials experimentation. The colored arrows show how feedback signals from open-ended experiments are routed to the agentic LLM, the materials science-tuned LLM, the pre-trained LLM, and their corresponding task modules. Different colors of arrows indicate the sources of the feedbacks. For example, if general-purpose pretraining data does not contribute positively to the ultimate goal of novel and useful materials discovery, adjustments should be made by revising the pre-training tasks and corpus or fine-tuning the pre-trained LLM accordingly to better align model knowledge with the ultimate objective of discovering novel materials. Dashed arrows denote the forward information flows. Note that humans are not only responsible for monitoring the open-ended experiments, but also co-improve with the agentic LLM.

Materials Domain

AI Methods

DiscoveryOriented?

Related Survey

Notable Features Van et al.11 Broad Broad No Task-based taxonomy Madika et al.12 Broad No LLMs Yes

Physics-informed, and responsible AI

Materials discovery for crystalline and molecular materials

Pyzer-Knapp et al.13

LLMs for materials science

LLMs Yes

Peivaste et al.14 Broad Broad No Emphasis on data Jiang et al.15 Text-centric Text-centric Yes Literature mining

LLMs augmenting human researchers

Lei et al.16 Broad Broad No

End-to-end and pipeline-centric perspective

Ours Broad Broad Yes

Table 1: Comparison of our survey with other most recent ones on AI (especially LLMs) for materials science. “Broad” in the Materials Domain column means the survey covers a wide range of materials science tasks, workflows, and concepts. “Broad” in the AI Methods column means the survey covers most AI methods, including traditional machine learning, deep learning, and LLMs. “Discovery-Oriented” is defined as systems prioritizing closed-loop validation and outcome-driven objectives.

well on material-related benchmarks? Q3 (The End-to-End Gap) What capacities have been 133 developed, and how can we bridge the gaps in capability, optimization, and governance between 134 existing AI for materials science and an end-to-end autonomous materials discovery pipeline? 135

There exist surveys on relevant topics, but they differ from ours, which are summarized in 136 Tab. 1. In contrast, we present a unified pipeline-centric perspective for review and discus- 137 sion. Driven by recent enablers such as autonomous labs, agent frameworks, and foundation 138 models, this survey charts a practical roadmap from foundation models to autonomous, safety- 139 aware LLM agents for discovering novel and useful materials that satisfy real performance, cost, 140 and reliability criteria. Targeting both AI researchers and materials scientists, we bridge these 141 disciplines to show why a system-level paradigm shift is now both necessary and feasible. 142

## 2 Recent Progress of AI and LLMs 143

The evolution from traditional machine learning to LLMs has progressively recast AI from a com- 144 putational tool into an intelligent research collaborator capable of reasoning, generating hypothe- 145 ses, orchestrating simulations, and designing experiments17. In this section, we review recent 146 AI and LLM advances with prospective relevance to materials science—irrespective of current 147 deployment—to address Q1 in Sec. 1: How have general-purpose machine learning models 148 evolved from merely passive processors to agentic systems? Rather than a chronological his- 149 tory, we trace this evolution through four key stages that mirror the pipeline in Fig. 1: (1) predic- 150

Section 3

Section 4

Section 2

Reactive Tasks in Materials Science

Agentic Systems for Materials Science

Recent Progress of AI and LLMs

| | |
|---|---|
|0|25|
| | |
|0|24|
| | |
|0|23|
| | |
|0|19|
| | |

[Figure 21]

[Figure 22]

NanoMINER

LLM-Syn-Planner

[Figure 23]

[Figure 24]

[Figure 25]

MatterChat PolyLLMem

[Figure 26]

MaTableGPT MATMMFuse

[Figure 27]

[Figure 28]

[Figure 29]

MatExpert

[Figure 30]

MultiMat

DiffractGPT

MATLLMSEARCH

[Figure 31]

[Figure 32]

MicroscopyGPT

[Figure 33]

SteelBERT

[Figure 34]

AutoLab

[Figure 35]

[Figure 36]

SmileyLlama

[Figure 37]

MatBind MatVQA

MolVision

[Figure 38]

[Figure 39]

[Figure 40]

MatMCL

mCLM

MechBERT

2

[Figure 41]

[Figure 42]

AI-Scientist

[Figure 43]

MatNexus

[Figure 44]

MatKG

[Figure 45]

[Figure 46]

UniMat

PolyTAO

ChemOS

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

Eunomia ElaTBot

CrystalLLM Geo2Seq

MatGPT

CALMS

[Figure 52]

[Figure 53]

IBM-KEP MELT

[Figure 54]

[Figure 55]

[Figure 56]

SciAgent

ChatExtract

2

[Figure 57]

[Figure 58]

ChartT5

OpticalBERT

[Figure 59]

[Figure 60]

[Figure 61]

CoScientist

KEBLM

PolyBERT

[Figure 62]

MaterialBERT

[Figure 63]

MTencoder

[Figure 64]

SFBC

2

[Figure 65]

[Figure 66]

[Figure 67]

MolT5

BatteryBERT

ChemCrow

[Figure 68]

[Figure 69]

[Figure 70]

AlphaFold

MatScIE

MatSciBERT

[Figure 71]

[Figure 72]

Mat2Vec

SciBERT

2

- Figure 2: Technology tree of AI4Material science research. With the emergence of LLMs and agents, research on materials science initially focused on domain specific task, primarily concentrating on seperate reactive tasks. Subsequent research has delved deeper, gradually integrating more with agentic systems for materials science.

tive models trained on static datasets; (2) foundation models serving as programmable priors; (3) 151 post-training methods for objective shaping and controllability; and (4) agentic systems enabling 152 tool use and long-horizon rewards. As shown in Fig. 1, pre-trained foundation models gener- 153 ally initialize the agentic pipeline for materials science; consequently, pre-training choices shape 154 downstream capabilities and should be iteratively adapted using RL signals from real-world ex- 155 periments. 156

### 2.1 Deep Learning Revolution 157

Machine Learning Foundations From the late 1990s through the early 2010s, most early 158 work in machine learning and AI was based on feature engineering, establishing the first batch 159 of data-driven tools for materials science. Over this period, these early methods were not just 160 used in materials research; they quickly became important instruments for finding new materials 161 and predicting their properties. 162

There are three main types of machine learning, which have been applied to materials sci- 163 ence. Supervised learning techniques, such as support vector machines18, random forests19, 164 and shallow neural networks20, facilitated property prediction from labeled data, thereby creat- 165 ing quantitative structure-property relationships for expedited materials screening. Ward et al.65 166 introduced a general-purpose framework that combined a chemically diverse feature set with 167 ensemble-based models like random forests. Their framework worked well on both crystalline 168

Supervised learning: e.g., Cortes and Vapnik18, Breiman19, Anderson20,

Unsupervised learning: e.g., McQueen21, Khan et al.22, Abdi and Williams23, Maaten and Hinton24

Machine Learning Foundations §2.1

Deep Learning Revolution §2.1

Reinforcement learning: e.g., Sutton et al.25, Mnih et al.26, Schulman et al.27, Garcia et al.28, Konda and Tsitsiklis29

Architecture: e.g., Krizhevsky et al.30, Goodfellow et al.31, Kingma and Welling32, Computation33

Deep Learning §2.1

Training: e.g., Srivastava et al.34, Ioffe and Szegedy35, Abadi et al.36, Paszke et al.37

Attention Revolution §2.2

e.g., Vaswani et al.38, Devlin et al.39, Brown et al.40

General-purpose: e.g., Bommasani et al.41;

Foundation Models §2.2

Science-specific: e.g., Beltagy et al.42, Jumper et al.43;

Recent Progress of AI §2

Transformer and Foundation Models §2.2

General-purpose: e.g., Wei et al.44, Borgeaud et al.45, Kaplan et al.46

LLMs and Reasoning §2.2

Materials-specific: e.g., Yu et al.47, Wei et al.48, Kim et al.49

General-Purpose: e.g., Li et al.50, Radford et al.51, Li et al.52

Multimodal Integration §2.2

Materials-specific: e.g., Tang et al.53, Choudhary54, Adak et al.55

Novel Architectures §2.2

e.g., Gu et al.56, Gu and Dao57

Emerging Paradigms §2.3

e.g., Ouyang et al.58, Radford et al.51

LLMs and Agents §2.3

Rise of Agentic System §2.3

e.g., Yao et al.59, Schick et al.60, Gravitas61

Recent Advances in Agents §2.3

e.g., Anthropic62, Shaw63, Jiang and Karniadakis64

###### Figure 3: Taxonomy of recent progress of AI and LLMs.

and amorphous systems, showing how powerful systematic feature design can be in speeding 169 up the discovery of new materials. Unsupervised learning techniques, such as clustering (e.g., 170 K-Means21, DBSCAN22) and dimensionality reduction (e.g., PCA23, t-SNE24), aided materials 171 discovery by enabling pattern recognition, phase identification, and feature extraction from high- 172 dimensional data66. Both supervised and unsupervised learning algorithms enabled the initial 173 requirements of the discovery cycle, i.e., correlating structures to properties. 174

To meet the need for autonomous materials discovery, RL techniques (e.g., Q-Learning25, 175 Deep Q-Networks (DQN)26, Proximal Policy Optimization (PPO)27, Model Predictive Control 176 (MPC)28, and Actor–Critic methods29) were introduced to interact with an environment and learn 177 from the consequences of their actions by optimizing reward functions over sequences of deci- 178 sions. These methods make it possible to explore chemical space67 and to search large design 179 spaces in a more systematic way to identify materials with desirable properties68. These models 180 work together to form an important foundation for the deep learning architectures that emerged 181 later. While these classical and early RL methods provided the foundational logic for data-driven 182 research, they lacked the representation learning capabilities required to process multi-modal 183 experimental data. 184

Deep Learning The 2010s saw a shift away from feature engineering and toward represen- 185 tation learning. The availability of large-scale datasets and increased computational resources 186 powered the development of deep learning69,70. The 2012 ImageNet competition winner, AlexNet30,187 helped establish deep learning as a general-purpose machine learning tool for real-world prob- 188 lems. In materials workflows, this period matters less as a milestone in ”AI history” and more as 189 the point when learned representations started to replace manually designed descriptors65,71. 190 This shift enabled scalable screening, surrogate modeling, and downstream decision-making in 191 materials discovery pipelines72,73. 192

After discriminative models proved useful and matured, the focus shifted to generative mod- 193 eling. Generative Adversarial Networks (GANs)31 made inverse design possible by training gen- 194 erator and discriminator networks against each other. Variational Autoencoders (VAEs)32 had 195 efficient inference and learning in probabilistic generative models with continuous latent vari- 196 ables. 197

Recurrent architectures such as Long Short-Term Memory (LSTM)33 and Gated Recurrent 198 Units (GRU)74 have made temporal modeling for reaction prediction, synthesis pathway optimiza- 199 tion, and materials degradation forecasting possible. Gradient-based optimizers, such as Adam 200 and SGD with momentum, together with regularization techniques like dropout34 and batch nor- 201 malization35 improved training stability and reliability. Tools like TensorFlow36 and PyTorch37 202 played an important role in simplifying model development. Especially their support for multi- 203 GPU training further accelerated progress and made it feasible to work with large databases of 204 materials75. As datasets continued to expand and recurrent architectures struggled with long- 205 range dependencies, the community has explored models better suited to handle global context. 206 This shift has led to the rise of the attention-based Transformer era. 207

### 2.2 Transformer and Foundation Models 208

The Attention Revolution The self-attention mechanism38 has reshaped sequence modeling 209 tasks since it was proposed in the Transformer framework. This made it possible to perform par- 210 allel computation and model long-range dependencies, which are important for understanding 211 chemical SMILES strings76 and scientific literature. The multi-head attention38 method enables 212 models to look at many representation subspaces simultaneously, capturing complex interac- 213 tions across them. 214

BERT (Bidirectional Encoder Representations from Transformers)39 introduced bidirectional 215 pre-training with masked language modeling and next sentence prediction in late 2018. This 216 enabled models to capture the complex, context-dependent relationships that characterize sci- 217 entific writing. GPT-340 demonstrated remarkable few-shot learning abilities with 175 billion pa- 218 rameters. Its performance suggested that scale, together with data, can give models surprising 219 flexibility, even without heavy fine-tuning. 220

Foundation Models and Scale-Up The term ”foundation models” was first used in41 to de- 221 scribe large models trained on a variety of data, typically using self-supervision at scale, that 222 can be applied to a wide range of tasks. This is because attention-based architectures made 223 it possible for these models to be built. These models represent a major shift in AI: their un- 224 precedented size leads to new abilities in language, vision, reasoning, and human interaction, 225 but they also increase the risks of homogenization, where flaws or biases in the foundation mod- 226 els spread to all adapted downstream systems. Task-specialized LLMs trained or adapted on 227 curated corpora (e.g., code77,78, math79,80, legal text81) exemplify the dual nature of scaling: en- 228 hanced domain proficiency frequently accompanies hallucination82 and challenges in calibration 229 and faithfulness83, alongside difficulties in controllability and verifiability84. 230

Field-specific foundation models have also emerged for scientific use. Domain-adapted sci- 231 entific foundation models tailor general priors to expert corpora and evaluation regimes. SciB- 232 ERT42 modified BERT for scientific literature by training it on 1.14 million papers from Semantic 233 Scholar. By improving skills like named entity recognition, citation prediction, and relation extrac- 234 tion, it made biomedical and computer science texts easier to understand. AlphaFold43 marked 235 a major advance in predicting protein structures with attention mechanisms for evolutionary data 236 and physical constraints. This showed how transformers can be used for complex scientific prob- 237 lems, motivating analogous designs in other scientific domains (including materials science). 238

LLMs and Reasoning Capabilities Recent research has focused on reasoning, knowledge 239 grounding, and tool-enhanced inference in addition to scaling. Chain-of-thought prompting44 240 adds the scientific reasoning that is essential for materials discovery. Memory-augmented archi- 241 tectures45 support the retrieval of external knowledge, hence enhancing access to specialist ma- 242 terials science literature and databases during inference. Scaling laws46 show that performance 243 improves when the number of parameters and data increases, following power-law relationships 244 up to the limits of the regime studied. This poses a challenge for materials research because 245 high-quality datasets are far less abundant than in many other domains. 246

In response to data scarcity, the community has explored strategies including the use of 247 synthetic data such as OC2085, pre-training on large unlabeled materials databases such as 248 the Materials Project86, and transfer learning from related scientific domains, including chemical 249 datasets such as QM987. The inherent tension between data requirements and availability con- 250 strains applications of foundation models in materials discovery to subdomains with sufficient 251 training instances. The improvement in the reasoning ability of LLMs enhanced multiple fields in 252 materials science research47, such as enhanced reasoning, which can help molecular structure 253 design49 and materials design48. 254

Multimodal Integration Cross-modal attention mechanisms have been used in multimodal 255 foundation models50,88 to integrate information from text, images, and structured data, which 256 is also useful for materials science, when workflows require fusing heterogeneous evidence 257 (e.g., literature, microscopy, and structure representations) into a single decision context. Vision- 258 language models (VLMs) like CLIP51 and BLIP52 use contrastive learning to enhance vision and 259

language representation alignment and thereby provide retrieval-ready joint embeddings that 260 can be repurposed with limited supervision. This ability has enabled tasks such as zero-shot 261 classification and text-image retrieval for the materials characterization study (e.g., rapid triage, 262 annotation, and cross-referencing of characterization images with textual descriptors), but reli- 263 ability typically depends on domain-specific calibration and evaluation. Flamingo89, KOSMOS- 264 290, and VILA91 are working on enhancing the video and 3D understanding abilities of the current 265 multimodal framework, by extending multimodal context length and enabling richer image–text 266 interleaving; however, robust 3D understanding for scientific data streams remains an open gap 267 for end-to-end discovery settings. 268

In addition to general-purpose multimodal models, there are also specialized multimodal 269 LLMs that have been adapted for materials science. These include MatterChat53, Microscopy- 270 GPT54, MolVision55, and PolyLLMem92. These models combine atomic structure data, text, and 271 images to make it easier for people and AI to work jointly on tasks such as discovering new ma- 272 terials and predicting their properties. However, the impact on discovery depends on how well 273 the system integrates aspects like tool use, uncertainty estimation, and closed-loop validation, 274 not just the model’s ability in isolation. 275

Novel Architectures Beyond Transformers To address the computational efficiency bottle- 276 neck of Transformer-based architectures on long scientific sequences, several alternative archi- 277 tectures have been proposed. Structured State Space Models (S4)56 addressed the limitations 278 of time-series materials data and molecular dynamics simulations by implementing parameter- 279 efficient architectures that facilitated training on exceptionally lengthy sequences (16K+ tokens). 280 Mamba57, a selective state space model with input-dependent state transitions and linear com- 281 puting complexity, can reach Transformer-level quality on language-modeling benchmarks while 282 scaling more favorably with sequence length. This efficiency is particularly relevant when mod- 283 els must handle long molecular strings (e.g., SMILES or SELFIES for complex molecules and 284 polymers), simulation trajectories from molecular dynamics or density functional theory (DFT) 285 relaxations, lab notebooks accumulated over long experimental campaigns, or tool outputs as 286 part of an end-to-end discovery workflow. The next paradigm shift moves away from passively 287 reading text and toward reasoning and tool usage skills, leading to autonomous agents. 288

### 2.3 LLMs and Agents 289

The recent innovations in AI have introduced what is termed “The Era of Experience”93. In this 290 view, progress is driven not only by fitting static human-generated corpora, but also by iterative 291 interaction with environments that provide feedback signals. Now the systems-based LLM can 292 complete complex experimental tasks to be an agent rather than a single network predictor. By 293 interacting with the environment/workflow, the agent can learn, improve, and adapt to real-world 294 tasks. 295

The AI field is currently transitioning from the “Era of Human Data,” where progress relies on 296 imitating finite, static datasets, to the “Era of Experience,” where agents achieve superhuman 297 capabilities by actively interacting with their environments and learning from the consequences 298 of their actions93. The relevance of this shift to materials science is practical: the interactive 299 nature can steer the whole pipeline towards the ultimate goal of discovering novel and useful 300 materials. 301

Just as general-purpose agents are now moving beyond human-privileged text interactions 302 to autonomous streams of experience, AI for materials science must evolve from passive models 303 trained on historical property databases to active AI Scientist94 agents that engage in closed- 304 loop discovery. By treating experiments and simulations not merely as data sources but as 305

interactive environments, such agents can generate their own high-quality data through RL, po- 306 tentially expanding exploration beyond human-selected candidates and accelerating iterative de- 307 sign–test–learn cycles. 308

Emerging Paradigms New paradigms, such as alignment and instruction tuning (e.g., Con- 309 stitutional AI95, RLHF58), have improved instruction following and controllability and can reduce 310 hallucinations in practice, although factuality is not guaranteed and typically depends on ground- 311 ing and verification, especially for scientific domains like materials science. These paradigms 312 are based on foundation models and can be viewed as post-training objective shaping on top 313 of pre-training. Few-shot96 and zero-shot learning51 have made it easier to adapt to tasks with 314 little data. This is important in fields where labeled datasets are limited by the high cost of ex- 315 periments and the need for domain expertise. In-context learning40,97 enables rapid prototyping 316 of new actions directly from stimuli, removing the need for task-specific fine-tuning. In order 317 to improve scientific decision-making rather than just isolated perception, models are becoming 318 more skilled at tool- and evidence-grounded reasoning across a variety of modalities, such as im- 319 ages98, crystal structures99, spectroscopic data, and experimental measurements. These works 320 suggest that agentic systems couple language-based reasoning with external actions (e.g., re- 321 trieval, simulation, and code execution). 322

Rise of Agentic Systems Enhanced agentic systems have been developed during the evolu- 323 tion of LLMs. In order to make decisions and improve grounding, agent loops combine reasoning 324 and action by utilizing external tools such as retrieval100, calculators and symbolic computa- 325 tion101, code execution or function calling59, browsers102, structured databases or enterprise 326 APIs103. Later frameworks like Toolformer60, AutoGPT61, and LangChain104 used these features 327 to add persistent memory, the ability to work with more than one tool at a time, and the abil- 328 ity to plan recursively. These works demonstrate that agentic systems can address problems 329 necessitating extended procedures, although consistent long-term performance continues to be 330 influenced by tool reliability, feedback design, and safety constraints. 331

Recent Advances in Agents In the second half of 2025, a wave of agents with thinking and 332 embodied action have emerged. Anthropic’s Claude Sonnet62 is a cognitive agent equipped with 333 enhanced reasoning, understanding, and tool-using abilities. This is a step toward AI systems 334 that are better at using tools and sustaining multi-step analysis in real workflows. It also supports 335 more scientific use by helping users structure questions, integrate evidence, and connect ideas 336 across domains. Gemini Robotics 1.5105 demonstrated transfer learning across different robotic 337 embodiments, advancing robotic agents beyond single-platform policies. Vision-language-action 338 (VLA) models highlight the value of explicit intermediate plans or reasoning steps that can im- 339 prove action. NVIDIA’s Cosmos reasoning model106 improves agent reasoning by integrating 340 physical priors and common sense constraints into robotic decision-making. Microsoft’s au- 341 tonomous agents63 demonstrate that AI agents have great potential to orchestrate complex, 342 multi-step tasks on behalf of users. 343

Through the above improvements, multiple agents with different skills can collaborate to build 344 agent ecosystems capable of completing complex tasks. Recent systems demonstrate that 345 AI can function as a powerful research partner, able to read literature, generate hypotheses, 346 plan experiments, and design syntheses, rather than just making predictions107. Such agents 347 can retrieve external resources, call specialized tools, and coordinate downstream actions au- 348 tonomously61,108. 349

Integrating data-driven analysis, such as extracting useful information from large-scale sci- 350 entific literature, with formulating hypotheses and organizing experiments can be supported by 351

LLM-based agents. Such capabilities are discussed in detail in64. These features make it much 352 easier to sort data for tasks that only apply to a certain area. 353

However, end-to-end materials discovery is still limited by reliable grounding, constraint- 354 aware reasoning such as stability and synthesizability, and closed-loop validation with simu- 355 lations and experiments. Agents can also enable more independent, agent-driven research 356 pipelines that may accelerate discoveries by emphasizing system-level integration (feedback, 357 coordination, and objective alignment) rather than isolated model capabilities. 358

### 2.4 Pipeline-Centric Perspective 359

[Figure 73]

Figure 4: The gradual integration of general AI into the materials science workflow. This framework illustrates the progressive evolution of broad, pre-trained foundation models as they are adapted for scientific discovery. Rather than a sudden replacement, the diagram depicts how general capabilities are iteratively refined through domain adaptation and feedback loops, allowing general-purpose AI to be gradually and effectively applied to specialized materials challenges.

After addressing Q1, we discuss the interaction between the pre-trained LLMs and the agen- 360 tic LLMs, as shown in Fig. 4, from the end-to-end pipeline-centric perspective, explicitly treating 361 the pre-training stage as a controllable module whose data, objectives, and safety constraints 362 can be continuously revised in light of downstream discovery performance rather than as a 363 one-off static initialization. This framing makes pre-training an object of system design—subject 364 to iterative revision—rather than a fixed prerequisite that is optimized once and then taken for 365 granted. 366

It is generally believed that general pre-training corpora on which contemporary LLMs are 367 trained are essential to the downstream fine-tuned models; however, recent studies (e.g.,109) 368 suggest that such broad pre-training may not always be necessary and, in some cases, may 369 even be suboptimal for specialized scientific applications. Furthermore, reliance on indiscrimi- 370 nately scraped web-scale corpora increases the chance of poisoning attacks and incidental mis- 371 information, which can induce harmful or misleading behaviors even when benchmarks appear 372 unaffected110. Instead, if the ultimate objective of discovering novel materials is what we value 373 most, we should review the techniques presented before and enforce that every upstream design 374 choice—including corpus selection, pre-training objectives, and alignment procedures—is justi- 375 fied by its contribution to this end goal rather than by improvements on proxy language bench- 376

marks. More concretely, better pre-training should be defined by downstream discovery-relevant 377 behaviors (e.g., planning feasible syntheses and correctly handling hazards), not by generic 378 language metrics alone. Continuing the presentations in Sec. 2.2, we should require that these 379 architectures benefit the discovery of novel materials and not just improve the pre-training scores 380 on generic benchmarks, recognizing that even small fractions of poisoned or misaligned data can 381 strongly bias downstream scientific behavior. 382

To make this pipeline-centric stance operational, the pre-training stage must be embed- 383 ded within a feedback loop in which agentic systems operating over simulations and experi- 384 ments supply long-horizon reward signals that can, in principle, be propagated back to ear- 385 lier stages to revise corpus composition, sampling strategies, and alignment objectives when- 386 ever inherited behaviors of a general-purpose LLM systematically hinder materials discovery. 387 This suggests moving from a frozen pre-training corpus toward an adaptive, feedback-driven 388 pre-training loop where materials-science outcomes and safety considerations jointly determine 389 which segments of the original web-scale data are amplified, attenuated, or removed, and where 390 domain-specific corpora are dynamically expanded in response to identified capability gaps 391 along the design–synthesis–validation pipeline. Recent progress in LLM-based agent architec- 392 tures and tooling, as discussed in Sec. 2.3, makes such a pipeline-centric perspective increas- 393 ingly feasible, allowing the successful discovery of novel, experimentally validated materials to 394 serve as the primary reward signal guiding the entire system rather than surrogate benchmarks 395 alone. Crucially, this closes the evaluation loop: it ties what we optimize (training choices) to 396 what we ultimately value (validated discovery). 397

Influence functions111,112 offer one concrete mechanism for such backward credit assignment 398 by estimating how infinitesimal upweighting or downweighting of individual training examples 399 would perturb a model’s parameters and downstream predictions. Approximate influence esti- 400 mators111 based on scalable inverse-Hessian approximations have been demonstrated on large 401 LLMs, recovering sparse, interpretable sets of pre-training sequences that are most responsible 402 for particular emergent behaviors. From a pipeline-centric perspective, these tools can be re- 403 purposed to identify which general-domain documents and which MatSci-specific corpora most 404 strongly support or undermine key capabilities needed for discovery (e.g., robust reasoning over 405 phase diagrams, synthesis planning under realistic process constraints, or safety-critical hazard 406 identification), thereby informing which data sources to emphasize, de-emphasize, or excise in 407 subsequent rounds of pre-training. This turns data selection into a goal-driven hyperparameter, 408 rather than a one-time curation decision. 409

Analogously, an AI4MatSci pipeline could employ influence scores computed on discovery- 410 relevant behaviors, e.g., successful identification of synthesizable candidates, safe handling of 411 hazardous chemistries, or robust performance in long-horizon design loops. It can upweight 412 beneficial segments of both general and MatSci-specific data, while downweighting or removing 413 sequences that systematically induce hallucinated mechanisms or impossible synthesis condi- 414 tions. In this view, pre-training, domain adaptation, and instruction tuning collectively form a sin- 415 gle, influence-addressable memory that is continuously reshaped by reward signals originating 416 from agentic closed loops in simulation and experiments, rather than being optimized only once 417 against proxy language benchmarks that may fail to reveal poisoning or subtle misalignment. In 418 this sense, influence functions become not only interpretability tools but also prospective levers 419 for dynamically re-aligning the pre-training component with the ultimate materials-science objec- 420 tive of discovering novel, useful, and safe materials, while mitigating security risks introduced by 421 open-world data collection. At the same time, these general-purpose agents must be carefully 422 tailored to materials science to align with real workflows, safety regulations, and experimental 423 constraints, a refinement that will be developed in the next section. 424

## 3 Reactive Tasks in Materials Science from an AI Perspective 425

191 have emphasized the importance of leveraging AI to accelerate the discovery of advanced 426 materials technologies by integrating information and decision-making across the composition- 427 process-structure-property-performance chain. Having discussed recent AI progress in the pre- 428 vious section, we will now review the evolution of traditional AI for materials science here to 429 ultimately answer the second guiding question Q2 introduced in Sec. 1: Why are existing AI for 430 materials science systems not enough to meet the needs of autonomous materials discovery, 431 even though they score well on material-related benchmarks? 432

We will describe the progress and contributions of AI methods across predominantly reac- 433 tive materials science tasks. However, we must carefully interrogate this progress in light of a 434 pipeline-centric perspective. Benchmark performance on isolated tasks such as property pre- 435 diction or information extraction can be misleading proxies for true discovery outcomes when 436 viewed outside the context of an end-to-end discovery loop, where synthesis feasibility, safety 437 constraints, and experimental validation ultimately determine success. As shown in Fig. 1, con- 438 temporary materials-science-tuned LLMs serve as key building blocks of agentic systems and 439 can be further specialized to improve agentic performance; yet, optimizing individual modules 440 for high benchmark scores alone may not translate into more efficient autonomous discovery 441 without pipeline-level objective alignment and credit assignment from real discovery outcomes 442 back to upstream components. 443

### 3.1 Prediction 444

Applying LLMs to materials science marks a significant paradigm shift, moving predictive mod- 445 eling from structure-specific deep learning toward generalizable, knowledge-intensive represen- 446 tation learning16. However, most current deployments still operate in a largely reactive, task- 447 conditioned manner. Prediction in this context encompasses both quantitative forecasting of 448 continuous properties (regression) and categorical assignment (classification), applied across 449 diverse materials data modalities. We argue that the key innovation of applying LLMs for ma- 450 terials science lies in its dual capacity to process symbolic structural inputs (e.g., formulas192, 451 space-group/Wyckoff descriptors193, and other text-encoded crystallographic representations194) 452 and high-dimensional feature sets (e.g., Crystal–structure–derived geometric features195, DFT- 453 derived features196, foundation-model embeddings197). This capability not only helps the devel- 454 opment of predictive models with enhanced performance and multimodal adaptability but also 455 facilitates the assessment of their trustworthiness181. 456

##### 3.1.1 Regression Tasks 457

Regression tasks for materials predict continuous-valued physical properties, such as thermo- 458 dynamic stability198, mechanical stiffness199, and electrical and thermal characteristics200. This 459 is a crucial step in accelerating the search for and design of novel materials. In practice, these 460 tasks typically assume fixed property definitions and evaluation protocols, which can limit adapt- 461 ability in end-to-end discovery pipelines201. The effectiveness of these prediction models relies 462 on their contextually enriched embeddings, which are obtained from various representations of 463 the materials, such as crystal structure99, chemical composition202, or textual descriptions203. 464 Standard machine learning and GNN models have established strong benchmarks. However, 465 the deployment of LLMs has introduced improved capabilities for property prediction by utilizing 466 global context and multi-modal data16. Still, it has trouble adapting204 across domain shift, new 467 target properties, and constraints relevant to synthesis and processing. 468

Electronic Properties:e.g., Jin et al.113, Jin et al.114, Li et al.115

Mechanical Properties: e.g., Lee et al.116, Liu et al.117

Regression § 3.1.1

Thermodynamic Stability: e.g., Faber et al.118, Houchins et al.119

Thermoelectric and Thermal Properties: e.g., Ghosh and Tewari120, Korop and Prybyla121

Defect Classification and Experimental Automation: e.g., Bran et al.122, Boiko et al.123

Prediction § 3.1

Classification § 3.1.2

Hybrid GNN-Transformer Architectures: e.g., Du et al.124, Li et al.125

Advanced Methodologies §3.1.3

Multi-Task Learning (MTL) for Generalization: e.g., Caruana126, Prein et al.127

Uncertainty Quantification: e.g., Tavazza et al.128, Tavazza et al.128

Information Extraction §3.2.1

e.g., Kim et al.129, Huang and Cole130, Sierepeklis and Cole131, Kumar et al.132

Reactive Tasks in AI for Materials Science §3

Knowledge Graph §3.2.2

e.g., Wang et al.133, Court and Cole134, Venugopal and Olivetti135, Zhang et al.136

Mining §3.2

Database Automation and Annotation §3.2.3

e.g., Yan et al.137, Zhang and Stricker138, Guha et al.139

Structure Generation §3.3.1

e.g., Antunes et al.140, Gruver et al.141, Gan et al.142

e.g., Takahara et al.143, Pan et al.144, Cleeton and Sarkisov145, Fung et al.146

Inverse Design §3.3.2

Generation §3.3

Synthesis Route Generation

e.g., McDermott et al.147, Huo et al.148, He et al.149

§3.3.3

###### Figure 5: Taxonomy of reactive tasks in materials science from an AI perspective, Part 1.

Materials Discovery Process §3.4.1

e.g., MacLeod et al.150, Chitturi et al.151, Zhang et al.152

Simulation and AI-based Verification §3.4.2

Optimization and Verification §3.4

e.g., Fare et al.153, Merchant et al.154, Lan et al.155

Agent-based Closedloop Labs §3.4.3

e.g., Kulichenko et al.156, Abolhasani and Kumacheva157, Seifrid et al.158

Data Augmentation: e.g., Chanussot et al.159, Szymanski et al.160, Gibson

- et al.161 Multi-fidelity learning: e.g., Pilania

- et al.162, De Breuck et al.163

Data scarcity and Heterogeneity §3.5.1

Few-shot Learning: e.g., Guo et al.164, Qian et al.165, Zhang et al.166

Physics-informed Neural Networks: e.g., Chen et al.167, Fang and Zhan168, Zhang et al.169

Reactive Tasks in AI for Materials Science §3

Data and Knowledge §3.5

Knowledge integration §3.5.2

Structure-based Knowledge Integration: e.g., Jha et al.72, Wang et al.170, Chen et al.171

Ontology and Knowledge Graph Integration: e.g., Zhao et al.172, Fang et al.173, Fang et al.174

Pre-trained Model: e.g., Huang and Cole175, Zhao et al.176, Ock et al.177

Multimodal Scientific Data: e.g., Gupta et al.178, Chen and Ong179, Xing et al.180

Multimodality §3.5.3

Multimodal Fusion: e.g., Moro et al.181, Pai et al.182

Sparse and Closed-form Models § 3.6.1

e.g., Ouyang et al.183, Xu and Qian184, Muthyala et al.185

Attention and Graph Explainers §3.6.2

Explainability §3.6

e.g., Jain and Wallace186, Kotobi et al.187, Das et al.188

Physicsinformed Interpretability §3.6.3

e.g., Batzner et al.189, Martonova´ et al.190

16

Figure 6: Taxonomy of reactive tasks in materials science from an AI perspective, Part 2.

The performance of machine learning models critically depends on their input representa- 469 tions. This challenge was acutely evident in early models for periodic solids, which had difficulty 470 formulating representations that were invariant to arbitrary choices of coordinate systems or unit 471 cells while guaranteeing uniqueness205. Early approaches adapted representations from molec- 472 ular machine learning (e.g., the Coulomb matrix) to periodic systems by incorporating Ewald 473 sums or considering neighboring unit cells118. A paradigm shift was subsequently established 474 by GNNs through representing crystals as graphs with atoms and bonds as nodes and edges, 475 which allows models to learn from the local atomic environment through message-passing71. 476 More recently, Transformer-based architectures have built upon this graph-based foundation by 477 processing materials as sequences of tokens—whether atoms, sites, or textual descriptions—a 478 strategy that, by focusing on global context, more effectively captures long-range interactions and 479 symmetries than the locally oriented GNNs leading to improved modeling performance206. How- 480 ever, improved representations alone do not resolve long-horizon planning, feedback coupling, 481 or decision-level constraints that arise when predictors are embedded in closed-loop discovery 482 workflows. 483

Electronic Properties In current benchmark-driven screening workflows, much emphasis has 484 been placed on the prediction of fundamental electronic properties, especially the bandgap and 485 conductivity for the screening of semiconductors and functional materials207. This focus is partly 486 pragmatic: these targets are widely used as first-pass filters, but they can also over-represent 487 what is easiest to score on proxy benchmarks. Recent works based on Transformers have shown 488 a superior capability in capturing global crystal characteristics compared to previous successes 489 of GNNs at capturing localized atomic environments113. The CrystalTransformer model em- 490 bodies this advance by feeding in crystal structure information through a Transformer encoder; 491 specifically, this work starts with a concatenation of chemical and structural feature matrices, 492 and then processes the combined representation into a Transformer encoder. In the encoder, 493 the multi-head self-attention identifies complex, long-range correlations between all atoms in a 494 manner that is agnostic to their spatial proximity, enabling the model to learn highly context-aware 495 atomic embeddings114. Other variants, such as MatInFormer, have generalized this paradigm by 496 learning the ”grammar of crystallography” through the tokenization of space group information 497 and subsequently demonstrating the adaptability of this architecture for property prediction on 498 a range of datasets115. Increasingly, a key factor is the strategic use of linguistic input. For 499 example, if natural language descriptions of a crystalline solid are used, then high-accuracy pre- 500 dictions of properties such as bandgap can be achieved, which seems not possible for some 501 GNNs due to an inability to capture certain nuanced, global structural details such as space 502 group symmetry and Wyckoff positions208. The success of Transformer-based architectures in 503 the processing of both symbolic structural information and text suggests that they operate as 504 superior contextual encoders rather than purely topological encoders209. This allows them to im- 505 plicitly integrate learned chemical and structural knowledge, avoiding the limitations of localized 506 message-passing schemes. 507

Mechanical Properties The prediction of mechanical properties such as bulk modulus and 508 shear modulus is important in materials design with targeted strength, durability, and resilience210,211509. Various machine learning models have been widely adopted for predicting such mechanical 510 properties, greatly reducing their dependence on simulation methods that are computationally 511 intensive116. More recently, domain-specific LLMs for materials science can be specialized in 512 the predictions of complex mechanical properties, often by leveraging text-derived representa- 513 tions or hybrid text+structured inputs. For example, ElaTBot is an LLM for the prediction of 514 the full elastic constant tensor from text inputs117. Hence, the elastic tensor can be predicted 515

simultaneously with the determination of properties such as bulk modulus at any finite temper- 516 ature. Moreover, by incorporating general-purpose LLMs like GPT-4o and using the Retrieval- 517 Augmented Generation technique, its prediction accuracy can be boosted, indicating again the 518 power of combining domain-specific knowledge with the broad reasoning capabilities of general- 519 ist models when retrieval sources are curated and verifiable212. This significantly accelerates the 520 prediction process but, more importantly, can support early-stage inverse design by proposing 521 candidate compositions with targeted elastic properties—subject to the fidelity of the predictive 522 surrogate and downstream verification. 523

Thermodynamic Stability Among the most basic and important thermodynamic properties 524 is the formation energy, which dictates stability for a crystalline solid213. Its prediction is thus of 525 paramount importance in order for high-throughput screening to be performed on new, synthe- 526 sizable materials. Machine learning models have thus been employed to directly predict it from 527 crystal structures, with the key ingredient being sophisticated feature engineering to account for 528 periodicity in solids118. Subsequent deep learning models demonstrated the feasibility of mate- 529 rial property inference from merely elemental composition and symmetry classification119. 530

A particularly noticeable development in this domain has been the development of structure- 531 agnostic models, which predict properties from stoichiometry alone. Using a message-passing 532 architecture on a dense weighted graph representation of the chemical formula, the Roost frame- 533 work learns material descriptors in an explicit-structure-free way214. This is especially important 534 in the early stage of materials discovery when little to no crystal structural information is known. 535 Self-supervised pre-training methods have been shown to boost the performances of such mod- 536 els even further, with substantial improvements in the low-data regime215. Furthermore, LLMs 537 have been demonstrated to handle simple chemical formulas and predict complex chemical 538 properties (e.g., formation energy), significantly reducing the computational cost of materials 539 screening. 540

Thermoelectric and Thermal Properties Thermoelectric materials enable the conversion be- 541 tween thermal and electrical energy, making them crucial for waste heat recovery and solid-state 542 cooling technologies. However, the lack of large, structured datasets with relevant performance 543 metrics hinders the development of models for high-performance thermoelectric materials216. 544 LLM-based extraction approaches are increasingly used to mitigate this data bottleneck. Recent 545 work has shown that agentic, LLM-driven workflows autonomously extract the thermoelectric 546 and structural properties from thousands of scientific articles120. These intelligent agents can 547 parse text, tables, and captions to create large machine-readable datasets that couple perfor- 548 mance metrics with structural information such as the crystal class and space group121. This 549 automated curation pipeline can improve downstream regression by expanding training data and 550 improving schema consistency; however, it still primarily optimizes proxy prediction objectives 551 unless coupled to closed-loop validation and decision-making in the discovery loop. 552

##### 3.1.2 Classification Tasks 553

Classification is a fundamental task in materials science, referring to the automatic categorization 554 of materials into predefined classes based on their composition, structure, or other characteristic 555 data72,217. Classification is a crucial step in numerous downstream applications, from interpreting 556 experimental data to controlling material quality and accelerating the discovery of new materi- 557 als207. Meanwhile, because LLMs can effectively handle noisy experimental and computational 558 materials data and integrate data inputs from multiple modalities such as text, images, and nu- 559 merical data16,218, they are gradually becoming a general solution for classification problems. 560

Defect Classification and Experimental Automation The classification of defects—whether 561 vacancies, dislocations, or grain boundaries—is an important aspect of predicting materials per- 562 formance and ensuring safety and optimization of manufacturing yield. In practice, defect signals 563 may be derived from microscopy219, in-situ sensors220, or the operational setting ranges from of- 564 fline labeling to on-the-fly process monitoring221. LLM has broad application prospects in this 565 field, serving both static analysis tasks such as defect labeling and integration into closed-loop 566 systems to provide real-time decision support for researchers, thereby accelerating research and 567 development cycles122,123. At present, most deployments still treat classification as a modular, 568 reactive component; closed-loop autonomy typically requires additional planning, control, and 569 verification modules beyond classification alone. 570

In additive manufacturing, LLMs fine-tuned on process parameter data can predict defect 571 regimes (i.e. keyholing or lack of fusion) directly from natural language descriptions222. This 572 enables faster build parameter optimization while significantly reducing reliance on resource- 573 intensive simulations and physical experiments. Similarly, in materials characterization, conver- 574 sational LLM systems have been developed to support defect classification from Atomic Force 575 Microscopy (AFM) images. These typically involve a two-step pipeline whereby a vision-based 576 model performs an initial defect classification, and then an LLM acts as a conversational assis- 577 tant. In this latter role, the LLM contextualizes the classification based on queries from users 578 and experimental parameters and provides actionable feedback in natural language, such as 579 suggesting adjustments of scan speed to minimize artifacts223. More broadly, classification be- 580 comes substantially more valuable when embedded in an interactive workflow, but the classifi- 581 cation objective itself remains a proxy label unless the surrounding system optimizes end-to-end 582 discovery outcomes224. This evolution transforms the LLM from a passive classifier into an active 583 participant in the experimental workflow-capable of hierarchical reasoning linking sensor input to 584 corrective measures with reduced reliance on continuous expert supervision16. Finally, beyond 585 classification, LLM-driven agents are being developed that automate the complete scientific dis- 586 covery process, from hypothesis generation225 to experimental design and data analysis123. 587

##### 3.1.3 Advanced Methodologies 588

To pursue more robust and generalizable LLMs for materials science, the adoption of deep learn- 589 ing methods has shaped both model architectures and training strategies. 590

Hybrid GNN-Transformer Architectures While GNNs perform well in modeling local atomic 591 environments, their main drawbacks are related to modeling long-range interactions and a well- 592 known issue of over-smoothing in deep architectures226. Transformers specialize in the oppo- 593 site: global dependencies with high computational costs for large structures227. Hybrid GNN- 594 Transformer architectures seek to harness the strengths of both approaches228. 595

These hybrid models generally use a GNN to learn efficiently local structural features from 596 predefined graphs. The node embeddings produced by this step encode structure-aware infor- 597 mation and are treated as a sequence of tokens by a Transformer module. This latter component 598 then uses self-attention to infer long-range interactions within the material124. The hierarchical 599 method therefore overcomes the individual limitations of both architectures by leveraging their 600 combined strengths to afford even more accurate and scalable predictions of material proper- 601 ties125,229. This synergy thus allows for a rich representation of materials from the atomic to 602 the macroscopic scale. However, stronger representations alone do not guarantee improved 603 out-of-distribution reliability, motivating uncertainty-aware evaluation and closed-loop validation. 604

Multi-Task Learning (MTL) for Generalization The physical interdependencies between key 605 material properties (such as band gap, elastic modulus, and formation energy) make MTL a 606 very suitable strategy for building generalizable models230. By training a single model on these 607 related tasks simultaneously, MTL seeks to discover a unified feature space that inherently en- 608 codes the underlying shared representations consistent with physical correlations, resulting in a 609 more robust and generalizable representation and leading to better prediction performance126. 610 A practical limitation is negative transfer: performance can degrade when tasks conflict or when 611 task data are imbalanced and heterogeneous, as is common in materials datasets231. 612

Architectures such as the Multi-Task Pre-trained Transformer Encoder (MTENCODER) learn 613 general material representations from diverse properties127. By combining the signals from sev- 614 eral auxiliary activities, this method makes data more efficient and helps the models generalize 615 better to new tasks and materials. Thus, using MTL in transformer encoders creates strong 616 and transferable embeddings that are an important step toward flexible foundation models for 617 materials science232. From a pipeline-centric perspective, MTL performs internal credit sharing 618 across proxy tasks; aligning task selection and weighting with discovery-level objectives remains 619 an open systems challenge233. 620

Uncertainty Quantification for Trustworthy AI To make AI-driven materials discovery effec- 621 tive, we need a system capable of accurately measuring prediction uncertainty. This measure- 622 ment of uncertainty is very important for building trust in AI systems, directing active learning 623 strategies, and reliably finding samples that are outside of the distribution128. 624

In closed-loop discovery, uncertainty is also a decision variable (e.g., acquisition or stop- 625 ping). There are two main kinds of uncertainty quantification methods: aleatoric uncertainty, 626 which comes from noise in the data, and epistemic uncertainty, arising from limited data cov- 627 erage and model misspecification234. There are different ways to figure out prediction intervals 628 for material properties. Some of these are probabilistic methods like Gaussian processes, while 629 others are quantile loss functions and methods that directly model prediction error128. Having 630 good estimates of epistemic uncertainty is very helpful because they show parts of chemical 631 space where the model is not very certain. This directly benefits targeted data collection, which 632 means that future experiments or simulations could be guided by an active learning loop. This 633 makes the process of discovering new findings faster and uses fewer resources235. In materials 634 settings, uncertainty estimation is further complicated by multi-fidelity data (simulation vs. ex- 635 periment), distribution shift across conditions, and the compositional long tail—precisely where 636 na¨ıve confidence estimates can fail. More and more studies are now trying to determine how 637 much uncertainty there is in the outputs of LLMs. This is a necessary step for using them reliably 638 in important scientific work236. Finally, while uncertainty quantification can make proxy-task pre- 639 dictors more reliable, end-to-end discovery requires propagating uncertainty through multi-step 640 planning and experimentation decisions. 641

### 3.2 Mining 642

##### 3.2.1 Information Extraction 643

Information extraction in materials science focuses on converting experimental descriptions and 644 performance data from publications into structured formats suitable for data-driven research. 645 As a pipeline-critical bottleneck, extraction determines what becomes trainable downstream 646 (datasets, benchmarks, and ultimately the objectives that models optimize). Early works used 647 rule-based text mining systems such as ChemDataExtractor237, which automatically produced 648 databases of synthesis parameters129, battery compositions130, thermoelectric properties131, 649

and mechanical data132 from journal article. These approaches provided reproducible extrac- 650 tion pipelines but were constrained by fixed grammars and limited adaptability to variations in 651 scientific writing. 652

To address these limitations, subsequent research introduced statistical and representation- 653 learning approaches. Embedding models trained on large-scale materials corpora captured 654 latent chemical and physical relations through co-occurrence patterns203, while domain-specific 655 advances in named entity recognition and normalization improved terminology consistency238. 656 Building on these foundations, transformer-based models such as MatSciBERT178 and Mech- 657 BERT239 provided contextualized embeddings that greatly improved extraction robustness and 658 accuracy across diverse publication sources. 659

Research on LLMs for materials science, utilizing prompt engineering and fine-tuning, has 660 expanded the limits of extraction system. ChatExtract240 builds a conversational system with en- 661 gineered prompts and follow-up questions to identify evidence sentences, extract structured val- 662 ues, and self-verify, reaching high precision and recall suitable for database curation. Similarly, 663 MaTableGPT241, an instruction-tuned and table-aware model, jointly reasons over narrative text 664 and tabular data for quantitative extraction. da Silva et al.242 presented an automated pipeline 665 for extracting synthesis procedures for reticular materials from PDFs using prompt-based para- 666 graph classification. Wang et al.243 also explored LLM-augmented decision programs that link 667 extracted synthesis data to design workflows. At the same time, others combine extraction 668 and verification mechanisms to autonomously construct structural datasets from full-text litera- 669 ture120,218,244. Further, Ansari and Moosavi245 developed a chemist AI agent that extracts and 670 structures materials science data with chain-of-verification tool to create datasets. Collectively, 671 these efforts suggest a shift from one-shot parsing toward agentic extraction pipelines, and yet 672 extraction still functions largely as a reactive module unless it is coupled to discovery-level ob- 673 jectives and outcome-grounded evaluation. 674

##### 3.2.2 Knowledge Graph 675

As extraction quality improved, researchers began to integrate extracted entities and relations 676 into structured knowledge representations. Hybrid and semi-supervised methods complement 677 purely model-based extraction by building structured knowledge representations. Hybrid meth- 678 ods combine GNNs, structured ontology models and LLMs to extend knowledge graphs and 679 interpret predicted relations246. Semi-supervised pipelines have been applied to identify synthe- 680 sis–processing–property relations in alloy systems133. Machine learning combined with text min- 681 ing has enabled the reconstruction of magnetic and superconducting phase diagrams from liter- 682 ature134. These extracted relations feed into knowledge graphs such as MatKG and terminology- 683 based networks135,136, which integrate entities, parameters, and outcomes into a structured rep- 684 resentation for downstream reasoning across the composition–structure–property space. Prein 685 et al.247 proposed a ranking-based approach to inorganic synthesis planning that learns pre- 686 cursor relationships in a latent space, providing new insights for reaction network construction. 687 A key limitation is that KG coverage and correctness remain bounded by extraction noise and 688 ontology choices, and thus errors could propagate into downstream reasoning and planning248. 689

##### 3.2.3 Database Automation and Annotation 690

Large automatically generated corpora137 enabled standardized benchmarks for entity and re- 691 lation extraction. Together, these efforts shifted the extraction paradigm from rigid rules to 692 learned semantic representations. Frameworks like MatNexus138 and MatScIE139 provide mod- 693 ular pipelines to aggregate extraction outputs and align them with ontologies. CALMS249 pro- 694 posed linking LLM-based extraction with instrument control and database queries. Multimodal 695

extraction combining text and tables250 and literature-based reconstruction of the materials tetra- 696 hedron251 illustrated progress toward unified modeling of composition, structure, process, and 697 property. From a pipeline-centric perspective, automated corpora and benchmarks can accel- 698 erate iteration, but they can also freeze proxy objectives when benchmark definitions drift from 699 discovery outcome-grounded evaluation and continual dataset expansion and revision. In prac- 700 tice, rule-free and model-agnostic curation workflows now turn full texts into mid-sized, high- 701 precision databases with minimal bespoke coding and near-perfect precision–recall trade-offs, 702 making them ideal for rapid property-focused datasets. 703

### 3.3 Generation 704

The generative paradigm has also profoundly shaped AI for materials science. LLMs, in particu- 705 lar, can be employed for designing novel materials and synthesis methods, expanding materials 706 R&D beyond data analysis into proactive innovation. Key tasks include structure generation, 707 inverse design targeting desired properties, and synthesis route generation. 708

##### 3.3.1 Structure Generation 709

The task of structure generation in AI4Materials spans multiple scales and materials classes, 710 aiming to propose new candidate structures that are valid, stable, and potentially synthesiz- 711 able252. In practice, these criteria are often operationalized using surrogates (e.g., predicted for- 712 mation energy), which can diverge from real synthesis feasibility—motivating the pipeline-centric 713 emphasis on verification and closed-loop feedback. While early research focused on crystalline 714 solids—where periodicity, symmetry, and lattice constraints dominate—the field now extends 715 to polymers, porous frameworks (MOFs253/COFs254), amorphous phases such as glasses and 716 electrolytes, microstructures at the mesoscale, and even architected metamaterials. In these 717 fields, materials can be represented as matrices, voxels, or graphs, and generative models255 718 are used to generate realistic atomic structures. For discovery, the key differentiators are less 719 the model family than whether generation is coupled to evaluation in the loop. 720

For crystalline materials, a complementary research approach has emerged that represents 721 crystal structures as text sequences and generates them using language models. This language- 722 based paradigm treats structural information as a sequence of tokens, which makes it possible 723 to generate lattices and compositions in an autoregressive manner. Following this idea, Crys- 724 talLLM140 further highlighted the potential of sequential generation by applying autoregressive 725 language modeling to produce CIF file sequences directly. Gruver et al.141 further demonstrated 726 that fine-tuned LLaMA-2 models can directly generate stable inorganic crystal structures purely 727 as text, while achieving nearly double the generation speed that of diffusion-based baselines (i.e. 728 CDVAE256). More recently, MATLLMSEARCH142 showed that pre-trained LLMs can act as agen- 729 tic generators of crystal structures without additional fine-tuning. Their system autonomously 730 explores and optimizes materials by combining a pre-trained LLaMA-3.1 model with evolutionary 731 search and physical evaluation. This shows a shift in regarding LLMs as reasoning agents that 732 can iteratively search, evaluate, and improve candidate structures. Collectively, these works 733 mark a transition from generative sampling to search-with-verification, which is loser to our 734 pipeline-centric view. Unfortunately, most evaluations still rely primarily on simulated or sur- 735 rogate validators. 736

##### 3.3.2 Inverse Design 737

In materials generation using generative models, the objective is typically to produce new ma- 738 terials that follow the distribution of the training data. However, in practical materials discovery, 739 there is often a demand to identify materials that exhibit specific desired properties. MatAgent143, 740 proposes an agentic materials generation framework combining an LLM, a diffusion model, and 741 a property prediction model. In this framework, the LLM proposes candidate compositions and 742 iteratively refines them based on feedback from the property predictor, enabling the autonomous 743 generation of materials with desired properties. RL has also entered this space: to inverse 744 inorganic oxide materials design to target promising compounds using specified property and 745 synthesis objectives144. Because RL naturally supports long-horizon objectives, careful reward 746 and constraint design is essential to avoid optimizing proxy signals that do not translate to real 747 discovery outcomes. Deep dreaming solves the inverse design challenge for metal–organic 748 frameworks by directly optimizing string-based representations of MOF linkers within a unified 749 neural network to generate novel, synthesizable structures with targeted properties145. Mat- 750 DesINNe is a framework based on invertible neural networks (INNs) that learns both forward 751 and reverse mappings between design parameters and target properties, enabling generation of 752 candidate materials for specified property goals (e.g., band gaps in 2D materials)146. Poly- 753 TAO (Transformer-Assisted Oriented model) is a generative pre-trained model for polymers: 754 given a vector of target polymer properties, it generates polymer repeat unit SMILES with very 755 high chemical validity and accurately matches desired property values in semi-template or even 756 template-free settings257. These approaches share a common dependency on surrogate prop- 757 erty models and feasibility constraints, motivating closed-loop verification later in the discovery 758 pipeline. 759

##### 3.3.3 Synthesis Route Generation 760

Synthesis route generation refers to constructing feasible pathways to synthesize a target ma- 761 terial. In materials science, it covers various aspects, such as choosing suitable precursors, 762 suggesting reaction paths, and making predictions about key reaction conditions like time, tem- 763 perature, and reaction environment258. In practice, synthesis is often the dominant bottleneck: 764 feasibility and process constraints are what separate generated candidates from experimentally 765 validated materials. Graph-based methodologies were first applied to depict solid-state pro- 766 cesses and employed pathfinding to ascertain viable synthesis pathways147. Researchers have 767 also used machine learning to directly predict experimental conditions from data in the literature. 768 This has shown links between the stability of precursors and the temperatures or periods of syn- 769 thesis148. In addition to reactions, models can suggest sets of precursors based on how similar 770 the materials are, which makes recipes for inorganic synthesis more accurate149. At the sys- 771

- tem level, autonomous laboratories are already combining AI planning with robotics and active 772 learning. For example, A-Lab can design, run, and refine synthesis strategies for new inorganic 773 compounds iteratively again259. What makes such platforms non-reactive is closed-loop execu- 774 tion, where synthesis outcomes update models and trigger replanning, rather than ending at a 775 one-shot proposal. 776

Synthesis planning is a complex task requiring the collaborative optimization of advanced 777 algorithms such as deep learning, RL, and uncertainty-aware search260. The aforementioned 778 approaches reveal that synthesis planning is evolving toward a data-driven closed-loop system. 779 The systems must not only generate synthesis paths tailored to specific requirements but also 780 perform real-time updates based on synthesis outcomes261. These loops can supply discovery- 781 level feedback signals that support end-to-end credit assignment across upstream components. 782

### 3.4 Optimization and Verification 783

##### 3.4.1 Materials Discovery Process Optimization 784

As an instance of the optimization lab, a self-driving platform, Ada150, has been developed; an 785 AI-driven lab for the synthesis of palladium thin films that uses a Bayesian optimization algo- 786 rithm to autonomously optimize the process in a closed loop where objectives are measured 787 on experimental outcomes. By mapping the conductivity-temperature trade-off, the AI model 788 identifies the optimal low-temperature, high-conductivity formulations, which were later success- 789 fully validated in a scalable manufacturing process. Later, Chitturi et al.151 demonstrated an 790 autonomous material search platform by developing a multi-property version of Bayesian algo- 791 rithm execution (BAX), which transforms the user-defined targets into acquisition functions to 792 guide the autonomous targeted materials discovery. Reported evaluations emphasize efficiency 793 in locating target materials; for discovery-oriented settings, these should be interpreted through 794 outcome-level metrics (e.g., experimental sample efficiency), not benchmark-style scores alone. 795 Benchmarks showed that variants of BAX outperform the strong baseline methods in efficiently 796 locating target materials, but only in low-dimensional settings262. 797

SPINBOT152, an autonomous platform employing Bayesian optimization, optimizes perovskite 798 film fabrication based on photoluminescence feedback. Similarly, Chang et al.263 developed an 799 Autonomous Research System (ARES) utilized Bayesian optimization to accelerate carbon nan- 800 otube synthesis, achieving an 8-fold increase in the growth rate of carbon nanotubes, demon- 801 strating the effectiveness of closed-loop optimization. Besides Bayesian optimization, RLHF is 802 also used in M4olGen264 to achieve multi-objective optimization on property-constrained molec- 803 ular generation. 804

##### 3.4.2 Simulation and AI-based Verification 805

Fare et al.153 proposed a method by leveraging the multi-output Gaussian process that dynami- 806 cally fuses information from sources of varying accuracy and cost (e.g., simulations and experi- 807 ments). The optimization task is driven by Targeted Variance Reduction with Expected Improve- 808 ment (TVR-EI), a multi-fidelity Bayesian algorithm, verified by performing computational simula- 809 tions. More broadly, verification in materials discovery is inherently multi-fidelity (approximate vs 810 accurate), so pipeline-centric systems must allocate verification budgets adaptively. Merchant 811 et al.154 demonstrated Graph Networks for Materials Exploration (GNOME) for large-scale ma- 812 terials discovery by combining GNNs with Density Functional Theory (DFT) simulations. 813

For the autonomous investigation of the mechanism of catalytic reactions, Lan et al.155 ap- 814 plied deep RL methods by developing a high-throughput framework coupled with first-principles 815 DFT simulations. More recently, Wen et al.265 showcased the integration of Cartesian Atomic 816 Moment Potential (CAMP), a machine-learned interatomic potential (MLIP) built entirely in Carte- 817 sian space, in closed-loop simulations, as the next generation MLIP primitive. They have demon- 818 strated that CAMP offers systematic improvements through utilizing physically motivated atomic 819 moment tensors and their tensor products within a graph neural network framework to cap- 820 ture local atomic environments and higher-order interactions. From a pipeline-centric view, im- 821 proved MLIPs matter because they reduce verification cost and expand the feasible horizon of 822 simulation-driven loops, making longer-horizon, agentic planning more practical. 823

##### 3.4.3 Agent-based Closed-loop Labs 824

Closed-loop labs accelerate research through the integration of AI with automated and self- 825 driving experiments156–158,266,267. The “mobile robotic chemist” was presented by Burger et al.268, 826

employing a free-roaming robot, as an autonomous agent, to automate the researcher’s tasks. 827 The robot was tasked with searching for an improved photocatalyst, driven by a batched Bayesian 828 search algorithm exploring a ten-variable experimental space, and Gaussian process regression 829 was used to build a predictive model. The “Artificial Chemist”, a self-driven agent, was developed 830 by Epps et al.269 by coupling an autonomous microfluidic flow reactor with machine learning to 831 optimize quantum dot synthesis. Later Sadeghi et al.270 developed a self-driving fluidic lab by 832 integrating microfluidics, automation, and Bayesian optimization for the autonomous synthesis of 833 lead-free perovskite nanocrystals. Kusne et al.271 presented CAMEO, closed-loop autonomous 834 materials exploration and optimization, an autonomous agent designed to discover new inor- 835 ganic materials, driven by Bayesian active learning to explore the complex relationship among 836 the composition, structural phase, and properties of the material. The “A-Lab” was developed by 837 Szymanski et al.259, an autonomous platform for the synthesis of novel solid-state inorganic ma- 838 terials. The A-Lab platform employed natural language processing models trained on literature 839 to propose initial recipes, and optimization was carried out by ARROWS272, an active learning al- 840 gorithm that learns from previous experimental outcomes. Upon failure of synthesis, (ARROWS) 841 uses ab initio computed reaction energies to predict and optimize alternative reaction pathways. 842 More recently, Omidvar et al.273 presented an agent-based closed-loop platform for the syn- 843 thesis of perovskite solid solutions by combining ML-guided composition prediction, automated 844 solid-state synthesis, and high-throughput characterization. 845

LLMs have broadened self-directed materials exploration. In the realm of the application 846 of LLMs in chemical synthesis, Coscientist, an autonomous chemical research framework, has 847 been developed by employing GPT-4-based agents to orchestrate chemical experiments274. Ad- 848 ditionally, in materials characterization, LLMs are currently employed not only for experimental 849 design but also for instrument control and automation via code generation. For example, one 850 implementation demonstrated that an LLM could independently write executable scripts (e.g., 851 Python and LabVIEW) to control laboratory instruments from natural language prompts, show- 852 casing genuine hardware control with minimal human involvement275. In recent times, LLMs 853 have been used to achieve full laboratory autonomy by bringing together experiment design, in- 854 strument operation, data collection, and analysis into one system. More recently, LLM-centered 855 systems have been explored for tighter end-to-end loops that combine experiment design, instru- 856 ment operation, data collection, and analysis. Xie et al.276 described a comprehensive system in 857 which the LLM not only produced control code but also analyzed experimental outcomes and en- 858 hanced subsequent measurement circumstances, thereby establishing a closed-loop connection 859 between thinking and execution. These research efforts clearly show that we are moving toward 860 experimental autonomy based on natural language processing. In this new way, LLMs are not 861 just instruments for communication; they are also agents for coding, control, and reasoning that 862 connect human purpose with machine execution. 863

To integrate human experts with machine learning, Adams et al.277 developed a human-in- 864 the-loop Bayesian framework where expert feedback is incorporated as priors to guide auto- 865 mated phase mapping experiments. BOARS addressed the rigidity of predefined targets in au- 866 tonomous experiments by developing a human-in-the-loop workflow278. The optimization strat- 867 egy allows a human operator to dynamically define and refine the scientific objective during the 868 experiment by upvoting or downvoting real-time spectral measurements, successfully locating 869 the human-desired optimum in significantly less time compared to manual optimization. 870

### 3.5 Data and Knowledge 871

One of the long-standing challenges in applying AI to materials science is the scarcity and het- 872 erogeneity of material data and knowledge. This becomes more challenging for an end-to-end 873

agentic LLM for materials science, especially considering the costs associated with evaluating 874 the outputs in the real-world, never-ending experimental environment. 875

##### 3.5.1 Data scarcity and heterogeneity 876

The advancement of data-driven materials science is characterized by the collaborative endeav- 877 ors of all stakeholders to address the challenges of data scarcity and data heterogeneity279,280. 878 To solve this problem, systems like MatMiner281 and Robocrystallographer282 give us the tools 879 we need to turn different types of materials into standardized numerical descriptors that ML can 880 use. MatMiner has a full library of features that cover a wide range of attributes. Robocrys- 881 tallographer, on the other hand, is made to create interpretable features from crystallographic 882 data. 883

OPTIMADE283 provides a unified API enabling researchers to query diverse materials databases884 simultaneously, thereby partially addressing data heterogeneity issues. Ghiringhelli et al.284 pro- 885 poses a hierarchical metadata approach,FAIR, which stands for Findable, Accessible, Interoper- 886 able, and Reusable, advocating for a unified and streamlined philosophy for computational and 887 experimental data management. 888

The LeMatTraj project285 directly solves the problem of not having enough data by com- 889 bining nearly 120 million high-quality atomic configurations from the primary repository into a 890 single dataset. This is in addition to the infrastructure standards. This huge collection has the 891 high-quality, standardized data that is needed to develop strong machine learning interatomic 892 potentials. 893

Data Augmentation Data augmentation techniques are crucial for mitigating data scarcity 894 and heterogeneity in materials modeling, evolving from basic perturbation strategies to com- 895 plex physics-informed generative methods. Early studies, such as OC20159, supplemented DFT 896 datasets through random structural perturbations, short-timescale molecular dynamics simula- 897 tions, and electronic structure analysis. It then slowly changed into physics-informed augmen- 898 tation strategies: Szymanski et al.160 included experimental errors (lattice strain, crystal texture) 899 in the XRD pattern generation process, thus linking simulations to experiments. Gibson et al.161 900 proposed a physically motivated, computationally cheap perturbation technique that augments 901 training data to improve the accuracy of unrelaxed structure prediction by 66% and directly ad- 902 dressing the domain mismatch problem in crystal structure prediction. Based on this idea, Dinic 903 et al.286 presented an ML model capable of predicting the crystal energy response to global strain 904 by using available elasticity data to augment the dataset, which greatly improved energy fore- 905 casts for structures that were not perfectly straight. In addition to physics-motivated methods, 906 generative techniques that produce high-quality in-silico data to augment the original dataset 907 have developed.Chung et al.287 proposed a GAN-based data synthesis method to produce spec- 908 tral samples to address the class imbalance problem, thereby enhancing the performance of the 909 ML model trained on the augmented dataset. MatWheel288 introduced a framework that trains 910 the materials property prediction model using the synthetic data generated by the conditional 911 generative model, showing its potential in addressing extreme data-scarce scenarios in materi- 912 als science. 913

Multi-fidelity learning Due to the trade-off between cost and accuracy, materials science in- 914 volves measurements and simulation data with varying degrees of precision. To enable the fu- 915 sion learning of existing data with different levels of accuracy, researchers have proposed multi- 916 fidelity learning. Its objective is to maximize the utilization of existing data for high-accuracy 917

model predictions while minimizing additional computational and human resource costs. Pila- 918 nia et al.162 introduced this approach using co-kriging to fuse low-cost, low-accuracy DFT cal- 919 culations with sparse high-fidelity data, achieving accurate electronic band gap predictions at 920 significantly reduced computational cost. Further, MODNet163 replaced Gaussian process re- 921 gression with DNN while learning about the differences between high- and low-quality values, 922 which shows significant improvement in the results compared to learning on the sole high-quality 923 experimental data. In contrast, MD-HIT289 addresses the redundancy problem inherent in multi- 924 source datasets, employing greedy incremental algorithms to filter overlapping samples and pre- 925 vent overestimation of performance, thereby enabling more realistic evaluation and improved 926 generalization to out-of-distribution materials. 927

Few-shot Learning Few-shot learning, which is training a model on a small number of labeled 928 examples, can help models quickly learn new tasks, especially when there is not a lot of good 929 training data available. Meta-MGNN164 was the first works creating meta-learning frameworks 930 for predicting molecular characteristics. This framework allows models to quickly adapt to new 931 attributes with training on a few instances, and also use self-supervised learning from molecular 932 structures to make up for the lack of labeled data. AttFPGNN-MAML165 built on this by com- 933 bining hybrid molecular representations fingerprints and graph neural networks with ProtoMAML 934 meta-learning. This made it easier to adapt to new prediction tasks. Most recently, MolFeS- 935 Cue166 moved the field forward by adding dynamic contrastive loss functions that let pre-trained 936 models learn useful representations from very unbalanced and small amounts of training data. 937 It addresses the challenge of data scarcity in drug discovery and materials research. 938

##### 3.5.2 Knowledge integration 939

Knowledge integration in AI4MS refers to using relevant knowledge from materials science to 940 guide the design and training of AI models, thereby improving the accuracy, interpretability, and 941 generalization ability of the models. 942

Physics-informed Neural Networks(PINNs) Physics-informed neural networks (PINNs) are a 943 type of general-purpose function approximator that can embed knowledge of any physical laws 944 governing a given dataset into the learning process. Chen et al.167 employs PINNs for the solu- 945 tion of inverse scattering problems that retrieves the effective permittivity parameters of a number 946 of finite-size scattering systems in photonic metamaterials and nanooptics. Fang and Zhan168 947 applied deep physics-informed neural networks for electromagnetic metamaterial design, which 948 can be utilized in dealing with various practical problems such as cloaking, rotators, concen- 949 trators, etc. Zhang et al.169 employ PINNs for defect characterization in mechanical materials, 950 using equilibrium PDEs and constitutive laws to identify internal voids/inclusions from boundary 951 displacement measurements. 952

Structure-based Knowledge Integration ElemNet72 and CrabNet170 only use the elemen- 953 tal composition of the material as input. They automatically learn chemical relationships and 954 composition-property correlations between elements based on chemical relationships in the pe- 955 riodic table, without requiring any manually designed descriptors. In addition, GNNs such as 956 MEGNet171, ALIGNN290, ROOST202, and GnoME154 model molecules and crystals by treating 957 atoms as nodes, bonds as edges, and selectively adding global features such as temperature 958 or pressure. This approach allows deep learning algorithms to make very accurate predictions 959 of features such as formation energy, band gap, and elastic modulus. Based on this general 960

modeling, many new studies have incorporated additional prior knowledge of material struc- 961 ture features by introducing new encoding modules or learning objectives to integrate structural 962 knowledge: M3GNet179 introduced a many-body computation module to calculate the three-body 963 and many-body interactions; NequIP291 proposed E(3)-equivariant neural networks using geo- 964 metric tensors and equivariant convolutions based on tensor products with spherical harmonics, 965 which introduces feature encoding of the rotational and translational symmetries of the material 966 structure. Allegro292 removes atom-centered message forwarding while maintaining equivari- 967 ance by employing learned representations in iterated tensor products. 968

Ontology and Knowledge Graph Integration As discussed in Section 3.2, extensive knowl- 969 edge bases and knowledge graphs have been constructed from various sources, including aca- 970 demic articles and experimental analyses. This organized knowledge in databases, covering 971 multiple kinds of knowledge in materials science, can then be used to enhance model perfor- 972 mance in material property prediction tasks. Zhao et al.172 is specifically designed to address 973 the challenges of data heterogeneity in materials science. It uses knowledge graphs as a se- 974 mantic integration center and combines ontology mapping and automatic annotation to extract 975 information from academic articles. Both KCL173, KANO174, ESNet293 construct an element- 976 centric knowledge graph capturing intrinsic physical and chemical attributes of elements and 977 leverages it in a multimodal or contrastive learning framework to unify structural features with 978 semantic chemical knowledge for improved molecular and materials property prediction. Recent 979 innovations combine LLM with knowledge integration: MOFs-KG294 createed the first compre- 980 hensive KGQA benchmark specifically for materials science (MOFs) with 644 complex natural 981 language questions, demonstrating ChatGPT’s systematic application to domain-specific knowl- 982 edge graphs. 983

Pre-trained Model Pre-trained models have been demonstrated to be effective in natural lan- 984 guage processing, especially in data scarcity scenarios. Through training with limited data, pre- 985 trained models can be transferred to a specific task while showing strong performance. In mate- 986 rials science, pre-trained model are also widely used. Early work, like Mat2Vec203, learned word 987 embeddings in materials literature without human supervision, showing a surprising capability 988 to identify semantics in materials science text. This led to the development of material-specific 989 language models such as MatSciBERT178 and MaterialBERT295. They collect a large corpus 990 of materials science for continual training without changing the BERT architecture. Compared 991 to vanilla BERT, they show superior performance in identifying material-related property data 992 and phrases in the literature. This conclusion contributes to the emergence of various BERT 993 variations that were particular to certain materials subfields, such as BatteryBERT175 for battery- 994 related material, OpticalBERT176 for optical materials, PolyBERT296 for polymers materials, and 995 SteelBERT297 for metallurgy materials. UniMat177 propelled the field forward by merging mul- 996 tiple modes (atomic structure, XRD patterns, and composition) into strong joint embeddings. 997 This made it easier to find novel materials. SmileyLlama298 is further eolved with Direct Prefer- 998 ence Optimization(DPO) to optimize the structure generation. Recent ideas deal with structural 999 limitations: Huang et al.299 proposed a self-supervised and multimodal pre-training technique 1000 that can predict performance based only on stoichiometry, without explicit understanding of the 1001 atomic structure. MELT300 was the first to employ curriculum-based continuous pre-training, 1002 which used semantic knowledge graphs to convert models from broad-domain knowledge to 1003 material-specific concepts in a systematic way. 1004

##### 3.5.3 Multimodality 1005

Recent advances in applying artificial intelligence to materials research have demonstrated the 1006 benefits of domain-specific language models, graph-based predictors, and image analysis frame- 1007 works292,295,301. To effectively discover novel materials, agentic LLMs must process and under- 1008 stand multimodal data at each stage of the pipeline, not only as more inputs, but as a prereq- 1009 uisite for closing the loop, where each stage and component produces distinct modalities that 1010 must be jointly interpreted for decision-making59,302,303. Most existing improvements are lim- 1011 ited to single-model scenarios, where algorithms are designed for only a single type of data. 1012 However, scientific research often uses multiple types of input for representation, such as text 1013 reports, structural graphs, microscopy images, and spectroscopic signals. Each representation 1014 reveals the material’s properties from a different perspective. Integrating different types of data 1015 into a single analytical framework is challenging due to methodological and infrastructure issues. 1016 This section will explore these issues, first introducing the characteristics of multimodal scientific 1017 data, and then gradually delving into the various methods designed to integrate the data, with 1018 success defined in terms of cross-modal generalization, more reliable mechanistic inference, 1019 and improved closed-loop decision-making. 1020

Multimodal Scientific Data Materials science research increasingly relies on multiple data 1021 sources, each revealing different aspects of material properties, thus requiring tailored data 1022 analysis methods. Using domain-specific language models, textual information from publica- 1023 tions, patents, and experimental protocols has been put into order. MatSciBERT178 showed 1024 that it is possible to pull out entities and relationships from scientific literature. The MatSci-NLP 1025 benchmark304 set up standard tasks for synthesis action extraction, which made it possible to 1026 systematically compare and make further analysis on current methods. 1027

Image-based modalities provide further information at both mesoscopic and microscopic 1028 scales. There are big improvements in deep learning for microscopy, but a complete study305 1029 points out that some problems remains, such as failing to do 3D reconstruction, high reliance 1030 on synthetic data, and lacking enough annotated dataset. Another approach is through spectro- 1031 scopic and diffraction data. Interpretable XRD analysis180 shows that multi-phase quantification 1032 remain difficult, especially in the absence of simulation-based guidance. DiffractGPT306 ana- 1033 lyzes diffraction profiles in a novel way, treating them as sequence data, enabling autoregressive 1034 modeling methods to directly infer atomic structures. 1035

Recent workshops have enriched the research environment by providing multimodal datasets 1036 and benchmarks. The TDCM25 dataset307 gathers crystalline materials that change with temper- 1037 ature, along with aligned structural, image, and textual characteristics, which helps with research 1038 on phase transitions. Similarly, Takeda et al.308 showed a multimodal fundamental dataset that 1039 integrates SELFIES molecular representations, DFT-calculated properties, and spectral charac- 1040 teristics to improve the accuracy of property predictions. NanoMINER309 utilized language mod- 1041 els to extract structured knowledge on nanomaterials, connecting unstructured literature with 1042 structured graphs. These improvements show that multimodal resources are becoming more 1043 comprehensive, but they also show that there is still no common metadata or ontologies that 1044 can be used with each other. Because of this, it is still hard to reproduce results across different 1045 modalities. 1046

Beyond crystalline and bulk materials, molecular discovery introduces additional represen- 1047 tational challenges that require foundation models to be ‘multilingual’ across structural, physic- 1048 ochemical, and knowledge-based modalities. To accelerate scientific discovery, a foundation 1049 model must become “multilingual,” able to interpret the vast modalities of scientific data beyond 1050 text. This includes 2D310 and 3D311 structures described using geometric graphs, physicochem- 1051 ical properties (e.g., chirality, electrostatic potentials, and hydrophobicity), and dynamic data 1052

from molecular and quantum dynamics (MD/QD). It must also incorporate knowledge from liter- 1053 ature312 and knowledge bases313, such as images, charts314, and reaction pathways310. 1054

Some recent work proposed joint molecule and language learning315 and applied it to power 1055 conversion efficiency for organic photovoltaic (OPV) devices316. However, most molecule lan- 1056 guage models tokenize at the level of atoms and/or they do not infuse synthesis considerations 1057 up front, which precludes matching tokens with desired functions and/or leads to AI-generated 1058 new molecular targets that are not readily synthesizable. To address these challenges, the recent 1059 highly innovative, function- and synthesis-aware modular Chemical Language Model (mCLM)312 1060 proposed a unified multimodal architecture and a suite of specialized encoders to tokenize each 1061 modality into a biologically and chemically meaningful language that can be reasoned over. 1062 mCLM incorporates function- and synthesis-related knowledge into the molecule discovery pro- 1063 cess a priori. mCLM generates new data on demand in the forward direction by an AI guided 1064 closed loop, where a human scientist simply inputs a few natural language phrases or sentences 1065 describing the desired functions of a molecule and receives the molecular structure with the best 1066 synthesizable functional modules in return. 1067

However, some additional challenges remain. The self-attention mechanism in the Trans- 1068 former operates by allowing each token to aggregate information from other tokens based on 1069 semantic relevance38. This formulation is rooted in natural language processing, where word 1070 embeddings capture semantic associations and attention primarily aggregates unary informa- 1071 tion across tokens. However, such a design may not align well with materials discovery in 1072 the physical world, where governing mechanisms are often determined by explicit pairwise 1073 or relational interactions rather than semantic similarity317,318. For example, in organic photo- 1074 voltaic (OPV) materials, key performance-determining factors—such as donor–acceptor energy 1075 level alignment and balanced charge transport—are jointly determined by interacting molecular 1076 pairs319,320. This mismatch suggests that future research should explore attention mechanisms 1077 that explicitly encode binary or relational “mutual influence” between material entities, introduc- 1078 ing inductive biases more consistent with physical interaction mechanisms317,318. Besides, while 1079 unified embedding-based multimodal models often improve predictive accuracy, they frequently 1080 provide limited mechanistic insight, raising concerns that learned representations may reflect 1081 correlations rather than causal or physically grounded relationships321. 1082

Multimodal Fusion Recent methodological advancements have concentrated on developing 1083 unified representation frameworks that integrate multiple modalities into a single representa- 1084 tion space. MultiMat181 presented a self-supervised foundation model that simultaneously en- 1085 codes text, structural graphs, microscope pictures, and spectroscopic profiles, thereby augment- 1086 ing transfer learning for subsequent property prediction. MatBind322 built upon this by using 1087 contrastive learning aims to align density-of-states, crystal structures, and textual descriptions, 1088 achieving strong cross-modal retrieval. MatMCL323 also enhanced the robustness of the model 1089 when dealing with data that only has some modalities, proving that multimodal information is 1090 redundant and complementary, and that the model can maintain prediction accuracy even when 1091 one or more modal information is missing. 1092

Other solutions change the way language models work, enabling them to handle non-textual 1093 data. For example, DiffractGPT306 uses sequential modeling approaches to analyze diffraction 1094 spectra. This shows that methods originally designed for natural language can also be applied 1095 to crystallography. Hybrid systems that combine (GNNs, such as M3GNet179) with LLMs further 1096 demonstrate how atomic representations can be linked to contextual knowledge in text. There 1097 are also new efforts to set benchmarks. MatVQA301 establishes multimodal question answering 1098 using figures and text, whereas MaCBench324 consolidates band structures, microscope images, 1099 and literature, facilitating a systematic assessment of general-purpose multimodal models in sci- 1100

entific endeavors. Fusion models like MATMMFuse182 use multi-head attention to combine crys- 1101 tal graph embeddings with SciBERT textual encoders. This consistently makes it easier to predict 1102 formation energies and electrical characteristics. However, most current multimodal benchmarks 1103 emphasize recognition and cross-modal alignment and only partially evaluate multi-step scientific 1104 reasoning or closed-loop decision-making—the regimes most relevant to autonomous discovery. 1105

These strategies substantially help bring together different signals, but there are still certain 1106 problems that need to be fixed. Unified embeddings can improve forecasts, but they don’t help us 1107 understand how things work very well. This raises the possibility that the captured relationship 1108 might show correlations instead of causal ties. Using high-dimensional modalities like micro- 1109 scope images increases computational and data-engineering costs, which makes them harder 1110 to access. Evaluation protocols are still fragmented. For example, MatSci-NLP304 only covers 1111 text-based tasks, and there are no common benchmarks for multimodal reasoning or cross- 1112 modal generalization. TDCM25307 and MaCBench324 are two other efforts that show progress 1113 toward shared resources. However, there is currently no consensus in the sector about evalu- 1114 ation suites. Future research directions encompass the integration of physical invariances into 1115 fusion architectures and the establishment of standardized multimodal datasets featuring inter- 1116 operable ontologies, both of which are essential for ensuring that multimodal AI significantly 1117 advances scientific discovery and can be evaluated by its contribution to end-to-end discovery 1118 outcomes (e.g., experiment selection, validation throughput, and constraint satisfaction), not only 1119 predictive metrics. 1120

### 3.6 Explainability 1121

Explainable AI is becoming a key component of AI-based applications in materials science. This 1122 change is happening because the field increasingly requires models that go beyond forecast- 1123 ing to provide mechanistic insight, error diagnosis, and trustworthy decision-making325. This is 1124 considerably more important if we use autonomous agentic LLMs in real-world tests, where ex- 1125 planations can serve as verification artifacts that gate high-stakes experimental actions. There 1126 are no universally accepted criteria for evaluating interpretations among groups. Alvarez-Melis 1127 and Jaakkola326 came up with three dimensions: explicitness, faithfulness, and stability. Ex- 1128 plicitness examines the clarity and comprehensibility of the explanations, faithfulness assesses 1129 whether the relevance score accurately reflects the true determinants of the model output, and 1130 stability assesses the consistency of explanations derived from similar inputs. Other research 1131 also makes a distinction between plausibility and faithfulness. Plausibility refers to consistency 1132 with human thinking, while faithfulness refers to adhering to the internal thought process of the 1133 model327. In a comparable framework, Lipton328 delineated three tiers of model interpretabil- 1134 ity: simulatability (the entirety of the model is comprehensible to humans), decomposability (the 1135 individual components are interpretable), and algorithmic transparency (the learning process is 1136 comprehensible and reproducible). 1137

In this survey, we consolidate these perspectives into three domain-relevant axes for evaluat- 1138 ing explainability in AI for materials science: 1139

- • Physical validity: Do the explanations align with established scientific reasoning and phys- 1140 ical laws, making them coherent to human experts? 1141
- • Faithfulness: Do the explanations accurately reflect the internal thinking process of the 1142 model, rather than being a post-hoc approximation? 1143
- • Stability: Do similar or adjacent inputs produce consistent explanations, ensuring robust- 1144 ness to minor perturbations or noise? 1145

These axes provide a unified evaluation framework covering sparse closed-form models, atten- 1146 tion and graph explainers, as well as physics-informed methods. They will guide our subsequent 1147 discussion of interpretability methods. 1148

##### 3.6.1 Sparse and Closed-form Models 1149

In materials science, few explainable models are more prominent than those that generate 1150 sparse closed-form representations between input features and target properties. The SISSO 1151 family of algorithms183–185, symbolic regression methods applied for perovskite catalysis329 sci- 1152 entific machine learning with sparse regression for stability equations330, and dimension-cons- 1153 trained symbolic regression for band gap prediction331. Unlike post-hoc explainers, these meth- 1154 ods directly return human-readable algebraic formulas that may be considered candidate scien- 1155 tific laws or descriptors. This makes them particularly appealing in domains like materials sci- 1156 ence, where physical interpretability and mechanism discovery are paramount. However, these 1157 methods typically operate on curated feature sets and may struggle with raw, high-dimensional 1158 modalities, motivating complementary neural explainers discussed next. 1159

Sparse and closed-form models offer two advantages. First, they frequently experience 1160 high levels of faithfulness, because the symbolic expression itself is not a proxy, but a predic- 1161 tive model. Second, their emphasis on physically meaningful operators (e.g., ionic radii ratios, 1162 electronegativity differences) improves the plausibility and physical validity. Yet the shortcom- 1163 ings of these approaches include poor stability: tiny changes in input data or the feature pool 1164 may produce different symbolic forms, which raises questions about reproducibility and robust- 1165 ness across datasets. The more recent algorithmic progress achieved, e.g., i-SISSO (mutual- 1166 information prescreening) and VS-SISSO (iterative variable selection), partially mitigates these 1167 problems by improving search efficiency and descriptor stability. Sparse and closed-form tech- 1168 niques are the best on physical validity and faithfulness overall, but their Achilles heel is stability, 1169 particularly under noisy or heterogeneous data. This limitation is especially consequential in 1170 closed-loop discovery, where unstable descriptors can misguide experiment selection and slow 1171 down exploration. This limitation motivates complementary directions such as attention-based 1172 explainers and physics-informed neural networks, which we discuss in the following subsections. 1173

##### 3.6.2 Attention and Graph Explainers 1174

Whereas sparse regression methods provide intrinsic interpretability by expressing structure- 1175 property relations in closed-form descriptors183, most state-of-the-art AI models in materials sci- 1176 ence are deep neural networks, in particular GNNs and transformer-based architectures325,332. 1177 These models achieve strong predictive performance across a wide range of tasks but are of- 1178

- ten criticized as “black boxes”, since their internal decision logic is difficult to inspect directly. 1179 In response, recent work has introduced both embedded and post-hoc explanation strategies 1180
- to make such models more transparent by highlighting which atoms, bonds, compositions, or 1181 spectral features drive individual predictions325,332. 1182

Representative examples include CrabNet333, a compositionally constrained attention net- 1183 work whose attention mechanisms surface influential elemental constituents of a compound; 1184 GANN334, a graph attention networks for 2D materials property prediction that localizes impor- 1185 tance to specific atoms and bonds; CrysXPP188, which trains crystal GNNs under property- 1186 sensitive explanatory paths to connect predictions with interpretable substructures; and spectro- 1187 scopy-focused interpreters that generate attribution maps of spectra to identify peaks that con- 1188 tribute most to the output187. Together, these approaches move beyond raw accuracy toward 1189 more transparent materials models that can be interrogated by domain experts, even though 1190

important challenges remain in ensuring that the resulting explanations are faithful, stable, and 1191 aligned with underlying physical mechanisms. 1192

Attention-based and graph-based explainers have strengths in their capacity to provide lo- 1193 cal, instance-particular reasons, like determining which atoms, bonds or spectral peaks guided 1194 the most prediction332,335. This increases the physical validity of their measurements, as high- 1195 lighted elements tend to conform to established chemical intuition. Nevertheless, faithfulness 1196 is not always guaranteed: attention weights may not correspond to the model’s actual decision 1197 logic, raising the question of whether highlighted features are causally responsible or only corre- 1198 lated186. 1199

Stability still remains a hurdle, because inputs perturbed in small amounts or random seeds 1200 can modify the highlighted substructures or spectra335. There are multiple approaches, in- 1201 cluding integrated gradients336. It appears that the durability of these explanations in materi- 1202 als models has been studied, and the proposed robustness is explored by perturbation-based 1203 tests335. Thus, overall, attention- and graph-based explanation approaches are highly desirable 1204 for supplying domain-plausible, instance-level insights. They are great for alignment with human 1205 chemical reasoning (plausibility/validity) but suffer from challenges in faithfulness and stability. 1206 These limitations motivate hybrid approaches that have a combination approach: attention using 1207 perturbation tests or physics-informed constraints, leading into the next subsection on physics- 1208 informed interpretability. 1209

##### 3.6.3 Physics-informed Interpretability 1210

A third group of solutions provides intrinsic interpretability by embedding known physical princi- 1211 ples directly into the model design. Physics-informed methods constrain predictions to physically 1212 consistent families, which builds trust and usefulness in materials research325. They don’t rely 1213 on explanations that come after the fact. Symmetry-equivariant networks, constraint-based ar- 1214 chitectures, or the strict implementation of conservation and dimensional norms can all help with 1215 this189. For example, equivariant graph neural networks that take into account the translational, 1216 rotational, and permutational symmetries of crystalline systems189; models that are limited by 1217 physical invariants like energy conservation or stress–strain relationships190. 1218

The fundamental benefit of physics-informed interpretability is that it focuses on physical va- 1219 lidity: predictions follow the known structure of the issue space without making up correlations169. 1220 Physics-informed interpretability enhances faithfulness, stability, and safety by confining models 1221 to physically valid actions, acting as critical feasibility guards in closed-loop discovery. But these 1222 limits can make it hard to be flexible and generalize when the laws of physics are complicated or 1223 not fully understood. This means that we need hybrid approaches that combine clear physical 1224 rules with data-driven learning to make strong, agentic pipelines. 1225

### 3.7 Pipeline-Centric Perspective 1226

After covering Q2, we discuss the interaction between the materials science fine-tuned LLMs 1227 and the agentic LLMs as shown in Fig. 1. On the one hand, existing pre-trained LLMs lack 1228 plasticity337,338, focusing on how these two components jointly implement decision-making in our 1229 pipeline-centric framework Moreover, compared to the general pre-training data, MatSci domain- 1230 specific data, as discussed in Secs. 3.1, 3.2, and 3.3, is orders of magnitude smaller even if we 1231 apply techniques like data augmentation as discussed in Sec. 3.5.1. These observations suggest 1232 that most behaviors of fine-tuned models are largely inherited from the pre-training stage, making 1233 it difficult to reliably elicit the desired behaviors using only limited MatSci-specific fine-tuning. 1234

Stepping back across the previous subsections, it becomes evident that most current AI4Mat- 1235 Sci formulations are defined around narrowly scoped, mostly supervised tasks such as property 1236 prediction, information extraction, structure generation, and local optimization, each tied to be- 1237 spoke datasets that are limited in both quantity and curation quality compared to the web-scale 1238 corpora used for general LLM pre-training as discussed in Sec. 2, and are typically optimized 1239 in isolation from end-to-end discovery outcomes. Even when augmented by multi-fidelity model- 1240 ing, synthetic data, or automated extraction pipelines, these datasets remain sparse and noisy, 1241 making them at best partial proxies for the rich, open-ended discovery problem that materials 1242 science ultimately cares about, leaving a substantial gap between benchmark success and real 1243 experimental impact. A straightforward response would be to scale up the quality, diversity and 1244 quantity of supervised MatSci tasks and benchmarks. However, from an end-to-end, pipeline- 1245 centric perspective that explicitly targets the discovery of novel and useful materials under finite 1246 resources, over-optimizing models for performance on such intermediate tasks risks locking ef- 1247 fort into local improvements on fragile surrogates (e.g., scores on a particular property dataset) 1248 while neglecting whether these gains actually translate into faster, safer identification and vali- 1249 dation of new functional materials. In this view, the MatSci-tuned tasks surveyed in this section 1250 should be regarded as adjustable operators within a larger discovery pipeline, not as endpoints; 1251 their design, data, and evaluation criteria ought to be continuously re-examined through the lens 1252 of the ultimate discovery reward, so that limited high-quality MatSci data is allocated to those 1253 parts of the pipeline that most directly advance real experimental outcomes rather than merely 1254 inflating intermediate scores. Consequently, we argue that effort should gradually shift from 1255 reactive intermediate MatSci benchmarks, as discussed in this section, toward more agentic 1256 systems that explicitly close the loop with simulation and experiment, which will be developed in 1257 detail next. 1258

On the other hand, increasing plasticity of the pre-training or adaptation stage carries the risk 1259 of eroding fundamental safety and security properties. Materials-science-specific tasks cannot 1260 reasonably be expected to cover the full spectrum of normative constraints encoded in general- 1261 purpose pre-training corpora, so aggressive adaptation must be balanced against the need to 1262 preserve these foundational principles. From the pipeline-centric standpoint, the challenge is 1263 therefore to couple relatively scarce, noisy MatSci supervision with general-purpose pre-training 1264 in a way that steers the overall system toward the end-to-end objective of autonomous, safe 1265 materials discovery, without sacrificing the broad alignment and robustness inherited from the 1266 foundation models. 1267

## 4 Agentic Systems for Materials Science 1268

In this part, we try to address the guiding question Q3 in Sec. 1: What capacities have been 1269 developed, and how can we bridge the gap between existing AI for materials science and a fully 1270 end-to-end autonomous materials discovery pipeline? 1271

In the previous sections, we reviewed general AI (Sec. 2) and AI for materials science (Sec. 1272 3), the latter of which focuses on reactive, isolated tasks such as property prediction, extraction, 1273 and generation. While these specialized modules have achieved high performance on static 1274 benchmarks, they remain fragmented components that do not, on their own, constitute a dis- 1275 covery engine. From our proposed end-to-end, pipeline-centric perspective, true acceleration in 1276 discovery requires integrating these capabilities into coherent, goal-directed loops that can plan, 1277 execute, and learn from experiments. Accordingly, we now review the last advances in agentic 1278 systems for materials science, examining how they transition from tools that passively handle 1279 assigned tasks to active processors capable of closing the discovery loop. 1280

Active Cognition e.g., Deng et al.339, MacLeod et al.340

Contemporary Agentic Systems §4.1

e.g., Tom et al.341, Ghafarollahi and Buehler342, Choudhary and DeCost343

Autonomous Loops

Safe Exploration and Future Ecosystems

e.g., Jiequn et al.344, Deng et al.339

Hypothesis Generation §4.2.1

e.g., Eriksson et al.345, Qian et al.346, Kusne et al.271, Ramos et al.347;

Agentic Systems for Materials Science §4

Critical Thinking §4.2.2

e.g., Kahneman348, Tom et al.349, Volk and Abolhasani350, Gladstone et al.351

Scientist AI §4.2

Experiment and Simulation Planning §4.2.3

e.g., Qian et al.346, Dehghannasiri et al.352, Talapatra et al.353

Hypothesis Revision §4.2.4

e.g., Ribeiro et al.354, Ouyang et al.183

Natural Language Interface

e.g., Wang et al.355, Choi and Lee356

Scientific Collaboration Network

e.g., Ghafarollahi and Buehler357, Sim et al.358, Roch et al.359

Human–AI Collaboration §4.3

Trust and Accountability

e.g., Panapitiya et al.360

###### Figure 7: Taxonomy of recent progress of agentic systems for materials science.

### 4.1 Contemporary Agentic Systems in Materials Discovery 1281

From Passive Prediction to Active Cognition Over the past ten years, AI’s role in materials 1282 science has changed from being a passive tool for analysis to an active participant in discov- 1283 ery. Early research mainly focused on prediction with graph neural networks (e.g., CGCNN99, 1284 ALIGNN343) and pre-trained models (e.g., CHGNet339). They can learn the mappings between 1285 structures and properties from static datasets. These approaches showed that complex atomic 1286 relationships in materials could be encoded into high-dimensional latent representations. How- 1287 ever, these AI models still operated passively. They could only predict but not interact. Therefore, 1288 agentic materials discovery establishes a new paradigm: AI is redefined as an autonomous sci- 1289 entific partner that capable of independently generating hypotheses, planning experiments, and 1290 continuously learning through feedback loops274,361. 1291

An agentic system in materials research operates according to the principle of closed-loop 1292 discovery. Formally, let a material’s state be represented by a descriptor vector x ∈ Rn, with 1293 measurable properties y = f(x) obtained through experiment or simulation. Traditional su- 1294 pervised learning seeks to approximate f. In contrast, an agentic system defines a control 1295 policy πθ(a|s) that selects actions a (e.g., synthesis parameters, composition proposals) based 1296 on system state s (e.g., prior experimental results, model uncertainty) to maximize a long-term 1297 objective R = E[ t γtrt]. In materials settings, this reward typically balances experimental per- 1298 formance, resource cost, and safety considerations rather than only predictive accuracy. RL 1299 thus becomes a natural formalism for autonomous experimentation362. However, in practice, 1300 the state and action spaces are highly non-convex, expensive to explore, and embedded with 1301 scientific constraints such as thermodynamic stability or synthesis feasibility. Consequently, the 1302 success of agentic systems depends on the integration of uncertainty quantification, active learn- 1303 ing, and surrogate modeling to guide exploration efficiently363,364. This integration distinguishes 1304 fully agentic pipelines365 from simpler Bayesian optimization loops by coupling rich state repre- 1305 sentations. 1306

Autonomous Loops: Self-Driving Labs and Multi-Agent Hierarchies Self-driving laborato- 1307 ries (SDLs) play an important role in autonomous loops. An SDL combines robotic hardware 1308 for synthesis and characterization with AI models that predict outcomes, evaluate uncertainty, 1309 and iteratively plan the next experiment366. In these systems, the learning algorithm closes the 1310 “materials loop”: (1) Design and selection: propose an experiment or select programs in the 1311 database, (2) Execution: perform synthesis autonomously, (3) Evaluation: measure outcomes, 1312 and (4) Evolution: retrain the model on updated data. This cycle implements concrete credit as- 1313 signment from measured outcomes back to the models that propose and prioritize experiments. 1314 Studies in autonomous thin-film optimization367 and catalyst discovery341 have demonstrated 1315 that such loops can simplify experimental search spaces and preserve physical interpretability 1316 at the same time. AI’s job in autonomous loops shifts from passive reasoning to agentic cognition 1317 with the ability to autonomously generate and evaluate hypotheses. 1318

Recently, hierarchical and multi-agent architectures have been developed to extend this au- 1319 tonomy recently. For example, MatExpert368 utilized multiply stages to mimic the human mate- 1320 rials design expert’s workflow. In multi-agent system design, there is no longer just one single 1321 decision-making entity. The discovery process is decomposed into several specialized agents, 1322 such as Predictor, Planner, and Verifier. Those agents can communicate with each other via 1323 standardized protocols342. The Predictor utilizes neural networks (e.g., CHGNet339, ALIGNN343) 1324 to evaluate candidate materials, while the Planner uses Bayesian optimization or RL to choose 1325 the promising directions in the action space. The Verifiers are embodied by robotic automation 1326 and are capable of conducting experiments and sending the results back to the shared memory. 1327 These architectures enable modular expansion and enhance explainability. Each agent follows 1328

a well-defined scientific rubric and simulates the division of human labor within research teams. 1329

Safe Exploration and Future Ecosystems Agentic systems must also balance exploration 1330 and exploitation. Pure exploitation may converge to local optima because it continuously re- 1331 fines known successful materials, while unbounded exploration wastes too many resources on 1332 random new paths. Techniques such as upper confidence bound (UCB) sampling and Thomp- 1333 son sampling in Bayesian optimization frameworks369 provide theoretical guarantees for efficient 1334 exploration. Meanwhile, recent RL approaches integrate epistemic uncertainty from ensem- 1335 ble models370 to quantify the expected information gain of each new experiment. In materials 1336 discovery, this translates into a quantifiable trade-off between novel compositions and reliable 1337 performance. Such uncertainty-aware strategies embody the concept of “safe exploration” in the 1338 physical sciences371. 1339

Another important development is the emergence of simulation and experiment in hybrid 1340 simulation-experiment loops372. High-fidelity simulations, including density functional theory 1341 (DFT) or molecular dynamics (MD) can act as virtual laboratories, generating synthetic data 1342 to pre-train models or provide experimental priors. Frameworks like DeepMD344 and CHGNet339 1343 provide fast computation that makes it capable of running millions of virtual experiments be- 1344 fore a single synthesis is attempted. Agentic systems that combine simulation with experiment 1345 bridge the gap between computation and reality. Virtual agents can co-evolve with their physical 1346 counterparts, which is closely related to the concept of digital twins for materials discovery373. 1347

The future trajectory of agentic systems will likely involve scientific multi-agent ecosystems374, 1348 in which autonomous entities collaborate, compete, and refine hypotheses and ideas under 1349 shared goals. By combining LLMs for reasoning and documentation with RL planners and sim- 1350 ulation engines, these ecosystems could perform discovery, hypothesis explanation, literature 1351 grounding, and experimental justification in natural language375–377. Such “AI laboratories” will 1352 require governance mechanisms378 to ensure safety and reproducibility, but they are also capa- 1353 ble of compressing the scientific feedback cycle from months to hours. 1354

In summary, agentic systems represent the emergence of AI, robotics, and scientific reason- 1355 ing into a unified framework for autonomous scientific discovery. This trend shows a change 1356 from using models only to speed up predictions to using them as cognitive and experimental 1357 co-pilots that actively guide the discovery process. In this process, machines not only compute 1358 results but also design the path to discovery. The next subsection explores how this transition 1359 naturally leads to the concept of Scientist AI: systems capable of generating and testing scientific 1360 hypotheses beyond pattern recognition towards end-to-end discovery. 1361

### 4.2 Scientist AI: Beyond Data Fitting to Scientific Reasoning 1362

The new phase in AI for materials science goes from making accurate predictions to using AI to 1363 do scientific reasoning on its own. This means systematically acquiring, testing, and changing 1364 beliefs about the physical world349. Early deep learning frameworks such as CGCNN99 and 1365 ALIGNN343, discussed in Sec. 4.1, are capable of learning intricate structure–property mappings. 1366 However, they cannot articulate hypotheses, mechanisms, or explanatory frameworks349. 1367

In contrast, a scientist AI system94 aims to operationalize scientific reasoning. It indepen- 1368 dently formulates hypotheses, orchestrates virtual or tangible experiments, and updates its in- 1369 ternal models in response to the resultant evidence361.Instead of just guessing what attributes 1370 would happen with certain inputs, this kind of system keeps and improves a list of possible ex- 1371 planations for how structure, processing, and environment affect behavior349. It also looks for 1372 data that will help test and improve those explanations349. Previous research on robot scien- 1373 tists379 and self-driving laboratories380 illustrates aspects of this concept, wherein automated 1374

systems repetitively suggest experiments, gather data, and revise internal models accordingly. 1375 This sub-section delineates emerging methodologies—active learning, autonomous experimen- 1376 tation, symbolic inference, and literature-grounded reasoning—as approximating various ele- 1377 ments of a reasoning-centric workflow. Nonetheless, these elements have yet to be amalga- 1378 mated into a cohesive scientific methodology: hypotheses are infrequently articulated, causal or 1379 mechanical reasoning is limited, and the integration of priors across modalities remains fragile. 1380

##### 4.2.1 Hypothesis Generation and Search Over Scientific Spaces 1381

One of Scientist AI’s main skills is using priors, evidence, and uncertainty to recommend path- 1382 ways in the materials space349. Predictive models can inform practitioners what the attributes of 1383 a target will be based on certain inputs. However, making hypotheses demands working in orga- 1384 nized scientific domains where composition, processing parameters, mechanisms, and physical 1385 restrictions all matter349. Recent advancements in machine learning and autonomous testing 1386 exhibit significant potential for proposal generation; yet, they still lag far behind human scientific 1387 thinking350. 1388

Contemporary Bayesian optimization (BO) and active learning frameworks have exhibited 1389 robustness in suggesting useful experiments characterized by quantified uncertainty. These 1390 methods usually use Gaussian process surrogates or similar probabilistic models345. The goal 1391 is to find a balance between exploration and exploitation in high-dimensional design spaces346. 1392 In materials science, BO has facilitated the swift optimization of alloys, catalysts, and thin-film 1393 systems350. It helps reduce the number of trials by several orders of magnitude and allowing 1394 for closed-loop experimental workflows271. These systems partially implement the hypothesis- 1395 proposal phase of scientific reasoning by identifying promising areas inside the design space 1396 amongst existing uncertainty381. 1397

Generative models offer an alternative approach to hypothesis formulation. Variational au- 1398 toencoders, generative adversarial networks, and diffusion-based models have been utilized in 1399 molecular and crystalline design382, yielding candidate structures tailored to certain target prop- 1400 erties or design goals383. These methods increase the effective hypothesis space, allowing for 1401 the investigation of chemically valid yet previously unrecognized compositions and microstruc- 1402 tures. Recent research illustrate inverse-design workflows wherein generative models provide 1403 candidates that fulfill various objectives or constraints, including stability and performance thresh- 1404 olds, succeeded by evaluation using high-fidelity simulations or experiments382. 1405

LLMs add another way to come up with hypotheses. LLMs and science-specific variations, 1406 like SciBERT, can read a lot of scientific texts and make sense of them42. They can also 1407 find domain-specific entities and relationships and provide mechanistic narratives or qualita- 1408 tive design rationales384. Recent agentic systems that link LLMs to chemistry and materials 1409 tools demonstrate that these models can suggest reaction pathways, enumerate viable synthetic 1410 routes, and advocate for experimental parameter sweeps by utilizing patterns found in the litera- 1411 ture and outputs from external tools385. These systems are increasingly including explicit priors 1412 and limitations when used with retrieval and organized knowledge bases384. This makes it easier 1413 to come up with hypotheses that are based on current materials knowledge and makes it easier 1414 to integrate with that knowledge347. 1415

Even with these improvements, today’s systems still do not use hypothesis generation in 1416 a scientifically sound way. First, hypotheses are seldom articulated openly; the majority of 1417 approaches provide candidate points or structures without delineating the corresponding me- 1418 chanical assumptions, validity regimes, or expected failure mechanisms350. Second, generative 1419 proposals are frequently assessed solely via surrogate property predictors, exhibiting minimal 1420 linkage to falsifiable assertions or established theories, hence limiting their function in systematic 1421

hypothesis testing and refinement350. Third, constructing long-horizon hypotheses—like multi- 1422 step reasoning from mechanism to candidate design to experimental protocol—continues to be 1423 fragile and heavily reliant on limited training distributions or manually generated prompts386. 1424 Lastly, even though LLM-based agents seem to be able to combine literature and coordinate 1425 tools, their factual reliability, calibration, and grounding in domain knowledge are still problems 1426 that need to be addressed. This is shown in broader evaluations of foundation models387 and 1427 domain-specific LLMs for chemistry347. 1428

All of these changes show that AI systems may suggest candidate materials, designs, and 1429 mechanistic theories based on data and with an awareness of uncertainty349. Nonetheless, they 1430 have not yet produced organized, falsifiable hypotheses or engaged in systematic reasoning on 1431 alternative explanatory theories386. To improve this skill, generative modeling has to be more 1432 closely linked to physical limitations, clear hypothesis representations, and systematic uncer- 1433 tainty quantification within a larger agentic loop for autonomous materials discovery349. 1434

##### 4.2.2 Scientific Critical Thinking 1435

Unlike general-domain LLM reasoning, chemical reasoning must deal with uncertainty, con- 1436 founded and scarce datasets, and slow, costly experimental feedback349,350. Literature data 1437 is often noisy or conflicting, and learning frequently depends on imperfect proxy rewards from 1438 simulations, since reliable signals from laboratory experiments are scarce and costly349. At the 1439 same time, molecular discovery is inherently multi-objective, where improving one function can 1440 come at the cost of damaging another388. 1441

Human cognition is often described as two modes of thought: System 1 (“fast thinking”) and 1442 System 2 (“slow thinking”)348. System 2 is crucial for complex problems that require deeper and 1443 deliberate decision-making. Recent work on Energy-Based Transformers frames “thinking” as 1444 an optimization that not only feeds richer context but also assesses the plausibility of its own 1445 predictions and dynamically determines when to halt reasoning351. Future research should aim 1446 to develop a System 2–style thinking agent to anticipate the functions and interactions of gen- 1447 erated molecules in future physical states and encode these imagined outcomes as additional 1448 input, enabling deliberate reasoning and hypothesis generation351. The model should also gen- 1449 erate and evaluate alternative hypotheses, fill in missing contextual information, and connect 1450 disparate findings389. Rather than trusting all sources equally, it should assign weights to claims, 1451 hypothesize negative results, and actively resolve conflicting evidence389. To improve its causal 1452 understanding, the model should simulate counterfactual alternative molecular designs and their 1453 expected outcomes390, prioritize underexplored regions of chemical space through active learn- 1454 ing, especially those characterized by conflicting evidence or high uncertainty in counterfactual 1455 simulations388. 1456

##### 4.2.3 Experiment and Simulation Planning (Decision-Making Under Uncertainty) 1457

Another important skill for a Scientist AI is being able to plan experiments and simulations when 1458 things are uncertain349,350. In this situation, planning is choosing the next actions—new mea- 1459 surements, simulations, or adjustments to the process—that are thought to be the most useful or 1460 informative based on the present hypotheses, available data, and priors352. Experiment and sim- 1461 ulation planning, on the other hand, needs decisions to be made in order with restricted budgets, 1462 different costs, and safety or feasibility limits on which activities can be taken353. 1463

Modern materials informatics has created a wide range of tools for flexible experimental de- 1464 sign349. Active learning and Bayesian optimization frameworks utilize uncertainty-aware surro- 1465 gate models and acquisition functions to dynamically select fresh measures that either optimize 1466 information gain or enhance a specified property350. Optimal experimental design formalizes 1467

the challenge as a trade-off between elucidating the response surface and differentiating among 1468 competing models, frequently within the confines of limited resources352,353. These methods al- 1469 ready allow practitioners to decide which composition, processing condition, or microstructure to 1470 evaluate next based on a current surrogate and its uncertainty estimates. However, they typically 1471 optimize scalar objectives or predictive uncertainty rather than explicitly targeting hypothesis dis- 1472 crimination or mechanistic insight350. 1473

In self-driving laboratory (SDL) architectures, planning experiments and automating the lab 1474 have been closely linked. Closed-loop platforms for thin-film and semiconductor discovery com- 1475 bine robotic synthesis and characterization with BO-based planners to autonomously search 1476 process spaces and improve multi-step recipes271. Multi-objective and restricted Bayesian op- 1477 timization has facilitated these platforms in managing trade-offs among competing performance 1478 measures while adhering to intricate experimental limitations, thus enhancing Pareto fronts for 1479 conductivity, transparency, or mechanical properties150. Recent reviews combine these changes 1480 and show that SDLs can be used as a general framework for planning closed-loop experiments 1481 in chemistry and materials science349. From a pipeline-centric viewpoint, SDLs implement end- 1482 to-end credit assignment by directly linking model updates and planner behavior to experimental 1483 outcomes, as opposed to relying exclusively on offline proxy metrics349. 1484

Emerging LLM-based agents enhance experiment and simulation planning by integrating lan- 1485 guage models with laboratory control systems, computational tools, and data services. Chem- 1486 Crow122 and other chemistry agents show that LLMs with special tools can design reaction se- 1487 quences, call simulators, query databases, and suggest follow-up experiments based on in- 1488 termediate results. Autonomous research frameworks utilizing general-purpose LLMs385 have 1489 demonstrated the capacity to strategize and implement multi-step experimental campaigns, en- 1490 compassing the selection of conditions, coordination of robotic platforms, and iterative enhance- 1491 ment of protocols349. Recent research assesses LLM agents for the automation of particular 1492 experimental modalities, like atomic force microscopy, underscoring both the potential and the 1493 existing fragility of language-model-driven laboratory control391. These systems are starting to 1494 integrate symbolic thinking, protocol generation, and tool invocation into a single planning cycle. 1495

Even with these improvements, today’s experiment and simulation planners still fall short of 1496 expert scientists in multiple dimensions. Most Bayesian optimization and active learning frame- 1497 works focus on optimizing scalar objectives or surrogate uncertainty; they do not explicitly repre- 1498 sent or update structured hypotheses352. Acquisition functions are usually defined over design 1499 variables instead of explanatory models or mechanisms. Consequently, campaigns seldom aim 1500 to differentiate between rival mechanistic hypotheses, delineate phase boundaries with measur- 1501 able confidence, or conduct stress tests in areas of theoretical ambiguity. Ad hoc filtering is 1502 typically used to deal with practical problems like scheduling, equipment failures, maintenance 1503 windows, or safety limitations, instead of using a systematic decision-theoretic formulation349. 1504 LLM-based planners can create flexible protocols and coordinate tools, but they are still restricted 1505 by hallucinations, poor calibration of uncertainty385, and inadequate guarantees that they will fol- 1506 low safety or physical rules122. Achieving scientist-level experiment and simulation planning will 1507 probably necessitate the amalgamation of uncertainty-aware optimization, explicit hypothesis 1508 representations, constraint-aware scheduling, and verifiable control of automated platforms into 1509 a cohesive framework for comprehensive materials discovery. 1510

##### 4.2.4 Interpretation, Explanation, and Hypothesis Revision 1511

Within the Scientist AI system, the interpretive part of the reasoning process is what pits new data 1512 against old hypotheses and comes up with new ones that better explain the facts. Section 3.6 1513 discussed how to make AI explainable in materials science in general. It introduced physical 1514 validity, faithfulness, and stability as ways to evaluate explanations392. In this context, these 1515

axes measure how well explanations help revise hypotheses instead of just checking models: a 1516 Scientist AI must use explanations to figure out which hypotheses are still valid, which are clearly 1517 wrong, and where more data would best improve the set of hypotheses393 394. 1518

Post hoc explanation techniques, such as feature attribution (e.g., LIME354, SHAP395) and 1519 saliency analyses, are now commonly utilized in composition-based models, graph neural net- 1520 works, and other architectures392. In materials science, these methods show which structural, 1521 compositional, or processing features affect individual predictions and whether models seem 1522 to be based on chemically plausible patterns392 393. Attribution methods for graph-based mod- 1523 els that pinpoint importance to atoms, bonds, or subgraphs further link predictions to specific 1524 motifs and have been evaluated on tasks with known ground-truth functional groups or struc- 1525 tural units335. From the standpoint of hypothesis revision, these tools facilitate a constrained 1526 form of testing: novel data can be evaluated for coherence between emphasized features and 1527 the mechanistic narrative embedded in a specific model class; however, the explanations them- 1528 selves typically do not preserve or revise an explicit hypothesis set. 1529

As discussed in Sec. 3.6, sparse and closed-form models like SISSO and symbolic regres- 1530 sion are unique in the explainability landscape183 396. Within the framework of Scientist AI, their 1531 primary contribution is to furnish explicit candidate hypotheses as human-readable descriptors 1532 or analytical laws that are subject to direct examination, scrutiny, and modification. Descriptor- 1533 discovery workflows that incorporate SISSO-style algorithms or symbolic regression into active 1534 learning loops for electrocatalysis and other applications exemplify this potential: the system 1535 not only modifies parameters within a static model but can also transition between different 1536 functional forms as new data emerges397. AI Feynman and other general equation-discovery 1537 frameworks show that it is possible to get governing equations from data in idealized settings398. 1538 This suggests a way to make reasoning modules that work with and choose between symbolic 1539 hypotheses instead of just updating neural weights that are hard to understand. 1540

Even though these systems have improved, they still don’t fully meet the needs of Scientist 1541 AI when it comes to interpreting and revising hypotheses. Most methods discussed in Sec. 3.6 1542 mainly focus on local or model-centric explainability, and their faithfulness and stability when the 1543 distribution shifts or the dataset is biased are still limited392,394. Symbolic techniques are getting 1544 closer to clear, testable hypotheses, but they still only work in low-dimensional environments and 1545 need a lot of expert input on feature spaces and operator sets183,396. Emerging LLM- and agent- 1546 based systems can generate textual rationales, alternative mechanisms, or literature-grounded 1547 critiques in response to new data107,399, yet these explanations are seldom tied to a formal rep- 1548 resentation of a hypothesis set, nor evaluated on their ability to drive correct revisions of that set 1549 over multiple iterations of data collection. To reach scientist-level interpretation and hypothesis 1550 revision, it is therefore essential to move beyond the primarily diagnostic explainability criteria 1551 and methods of Sec. 3.6 toward explicit hypothesis tracking, uncertainty-aware selection among 1552 competing mechanistic models, and benchmarks that directly measure how effectively an AI 1553 system revises and refines scientific hypotheses within a closed-loop discovery process. 1554

### 4.3 Human–AI Collaboration in Scientific Workflows 1555

Existing molecular AI systems remain static learners trained on historical literature data and often 1556 fail to adapt to new experimental evidence or improve through real-world feedback. Specifically, 1557 future work should establish a closed-loop digital–physical workflow where a high-throughput, 1558 autonomous laboratory can synthesize AI-selected molecules with minimal human involvement400,4011559. Recent work312 proposed to position mCLM as a core agent of a human-in-the-loop autonomous 1560 laboratory, iterating through reasoning, proposal, synthesis, testing, and feedback from real- 1561 world synthesis and testing machines to enable never-ending improvement alongside human 1562

scientists, by leveraging LLM agents’ self-improvement402,403 and self-evolving404 capabilities. 1563

As artificial intelligence systems mature from tools of prediction to agents of discovery, a 1564 central question has increasingly been posed: how should humans and machines collaborate in 1565 the pursuit of scientific progress405? The further improvement of AI4MatSci will not be achieved 1566 only by algorithmic and computational power. It will also depend on the design of human–AI 1567 workflows that involve human intuition and domain insight alongside machine-scale reasoning, 1568 data integration, and hypothesis generation378,406. This convergence motivates the design and 1569 use of co-creative intelligence407, in which AI could serve as a scientific collaborator. 1570

Early examples of such collaboration have already affected the materials design cycle. In 1571 prediction and generation tasks, researchers use interpretable graph neural networks (GNNs) 1572 such as ALIGNN343 or CrabNet333 to provide explanations for learned structure–property re- 1573 lationships, facilitating scientific validation and hypothesis refinement. These models support 1574 a two-way interaction between human insight and model-driven suggestions. AI can highlight 1575 correlations that are hard to detect intuitively in high-dimensional feature spaces while human 1576 scientists can test and interpret their physical plausibility. Studies in explainable materials mod- 1577 eling show that transparent model architectures, such as attention mechanisms revealing atomic 1578 contributions or saliency mapping over crystal graphs, can perform scientific structural reason- 1579 ing from numerical output.406. Such methods are critical for enabling trust calibration, where 1580 researchers learn not to over-trusting and under-trusting408. 1581

LLMs further facilitate the interaction between AI and humans by introducing natural-language 1582 interfaces into the workflows. LLMs that are fine-tuned on domain-specific training resources 1583 such as experimental protocols and computational reports can effectively parse, summarize, and 1584 contextualize scientific results. Recent systems like MatGPT355 are capable of solving materials 1585 language processing tasks by conducting workflows, querying crystallographic databases, and 1586 helping experimental setups by giving suggestions in natural language356. By integrating multi- 1587 modal reasoning that combines textual, numerical, and structural embeddings, LLMs can also 1588 serve as adaptive documentation and knowledge retrieval systems. These systems can evolve 1589 with laboratory data streams. In this manner, human–AI collaboration is expansive, from mere 1590 task execution to the whole scientific lifecycle: from idea creation to experiment construction and 1591 execution, validation, interpretation, and publication409. 1592

AI systems are becoming more and more autonomous, which makes human scientists be- 1593 come less directly involved in low-level decision-making. This will reduce the participation of hu- 1594 man agency and makes co-creation more difficult in some scenarios. More and more AI systems 1595 like ChemOS359 can enhance the planning and execution of experiments in self-driving labs. In 1596 this case, human scientists should foucs more on higher-level management, ethical oversight, 1597 supervision, and conceptual innovation366. Interface design further contributes to more effec- 1598 tive collaboration, which involves designing interpretable feedback loops between algorithmic 1599 reasoning and human comprehension410. For instance, prospective interfaces can prevent con- 1600 fusion before it happens. An explainable user interface allows users to interactively explore the 1601 data and model sensitivity so they can predict what and why the AI will do. The AI decision can 1602 construct AI results as structured hypotheses, inviting human assessment to decide whether 1603 accept or deny them. These hybrid decision-making architectures align with the principles of 1604 human-in-the-loop AI, where automation and human expertise together produce more reliable 1605 outcomes than either could achieve alone378. 1606

A second new concept at this stage is scientific collaboration networks, which were originally 1607 developed to analyze collaborations among human researchers411 and has since broadened to 1608 include hybrid collaborations between humans and AI agents357. In traditional scientific collab- 1609 oration networks, nodes are scientists, and edges connect them if the scientists co-authored at 1610 least one paper jointly. Networks were constructed independently for separate fields for com- 1611 paring the disciplinary patterns. By contrast, nodes in human-AI collaboration networks are 1612

specialized agents (software modules) along with human supervision. The framing implicitly fa- 1613 vors human-agent teaming: humans define goals, agents create and refine hypotheses, and 1614 humans evaluate, execute, and validate results. This structure acts much like a human research 1615 team. Different members have different roles that work well together, and they talk to each other 1616 through shared interfaces to plan their work. 1617

From an epistemological point of view, the collaboration of human–AI technology signifies a 1618 paradigm shift in scientific conduct. Rather than knowledge being created solely through hu- 1619 man deliberation, traditional science cycle by hypothesis–experiment–analysis is bounded by 1620 human thought and speed of human communication412. Instead, co-creative workflows exist at 1621 the crossroads between symbolic reasoning on the one hand and data-driven discovery, turn- 1622 ing this cycle into a process of feedback loop through AI412. Transforming experimental data 1623 into organized form representations understandable to humans and machines, these systems 1624 collectively share an epistemic language that brings interpretability and reproducibility413. 1625

Collaborative frameworks also need to address issues of attribution, authorship, and bias in 1626 the future in order to address the trust and accountability issues414,415. Designing new archi- 1627 tectures and evaluation frameworks for trustworthy collaboration is becoming essential. Auto- 1628 Labs360 introduces a critic agent specifically designed to double-check the safety and feasibility, 1629 while GIFTERS416 tries to evaluate the AI part along dimensions such as generalizable, inter- 1630 pretable, fair, transparent, explainable, robust, and stable trustworthiness principles. The scien- 1631 tific community is beginning to deal with questions like: Who owns AI-generated hypotheses? 1632 How can we peer review AI-based discoveries? Efforts such as transparent model reporting, 1633 tracking data sources, clear documentation, and continuous monitoring are essential for keeping 1634 the scientific integrity of AI-assisted research417.Furthermore, applying fairness, transparency, 1635 and accountability principles into materials workflows is crucial for the responsible adoption of AI 1636 in both industry and academia418. 1637

Human–AI collaboration is reshaping scientific discovery, shifting AI from a purely predictive 1638 tool to an active partner. In materials science, interpretable models and large language sys- 1639 tems bring continuous feedback between human intuition and machine reasoning, integrating 1640 speculation, simulation, and analysis into one workflow. As autonomous frameworks orchestrate 1641 experiments, preserving human agency in the process, open, interpretable interfaces are essen- 1642 tial. New networks of human and AI agents also appear, showing this move towards co-creative 1643 science. To maintain this partnership and its importance, safety mechanisms, interpretability 1644 guarantees, and scientific alignment protocols are needed to assure that their progressively 1645 autonomous AI systems are accountable and trustworthy, and in line with human scientific ob- 1646 jectives. 1647

### 4.4 Pipeline-Centric Perspective 1648

After addressing Q3, we continue our discussion on the end-to-end pipeline-centric perspective, 1649 focusing more on the role of agentic LLMs in materials discovery. Yann LeCun has famously 1650 argued that “if intelligence is a cake, the bulk of the cake is unsupervised learning, the icing on 1651 the cake is supervised learning, and the cherry on the cake is reinforcement learning” (NeurIPS 1652 2016), thereby downplaying the centrality of RL relative to representation learning via large- 1653 scale pre-training. In contrast, Silver and Sutton93 argue that intelligence should fundamentally 1654 be grounded in interaction with the environment, thereby elevating the role of RL and other 1655 experience-driven approaches to a primary mechanism for building general intelligence. Rather 1656 than engaging directly with this broader conceptual debate, we instead focus on the concrete 1657 demands of materials science, where discovery is inherently interactive: hypotheses must be 1658 tested through simulations and (preferably) experiments, so learning through experience is less 1659

optional than it is in purely digital areas. 1660

Given the domain-specific uniqueness outlined in the preceding subsections, these consid- 1661 erations collectively indicate that an agentic system constitutes the more appropriate design 1662 choice. However, current agentic systems reviewed in Sec. 4—while representing significant 1663 progress—remain incomplete from an end-to-end, pipeline-centric perspective for two critical 1664 reasons. 1665

First, a lot of current work doesn’t use backward credit assignment all the way through the 1666 pipeline. By backward credit assignment, we mean linking the results of downstream discover- 1667 ies—both successes and failures—back to choices made earlier in the process, such as corpus 1668 selection, objective design, and model adaptation. This way, those parts can be changed un- 1669 der the same end-to-end pressure. As outlined in Sec. 2 and 3, training signals resulting from 1670 agentic successes or failures (such as unsuccessful synthesis attempts, inaccurate property pre- 1671 dictions, and breached physical constraints) can theoretically be transmitted backward to prior 1672 stages, including pre-training data curation, domain adaptation objectives, and instruction-tuning 1673 criteria. This backward signal is necessary for the influence-driven correction discussed in Sec. 1674 2.4, where failures in closed-loop discovery immediately tell us which parts of the pre-training 1675 corpus should be made louder, quieter, or eliminated. Without this feedback loop, agentic sys- 1676 tems stay separate from the basic representations and assumptions built into their models. This 1677 makes it harder for them to change and get better over long periods of time. 1678

Second, most existing agentic systems work mostly in simulations or on synthetic bench- 1679 marks, not in real experimental settings. Simulation-based agents (e.g., those employing DFT 1680 or molecular dynamics) offer significant prototyping opportunities; however, they obscure essen- 1681 tial domain constraints, including equipment limitations, measurement noise, synthesis variabil- 1682 ity, and the inherent irreducibility of real-world chemistry to computational models. As a result, 1683 agents that are only trained on simulated loops may find optimal materials that are stable in silico 1684 but impractical or too expensive to make in real life. Real agentic systems for discovering new 1685 materials must be closely linked to robotic labs, experimental feedback, and the messy realities 1686 of physical synthesis and characterization. Agents can only learn useful tactics with affordable 1687 costs that close the gap between simulation and reality if they have a strong pre-trained foun- 1688 dation formed in the upstream stages, with experimentation providing the corrective signal that 1689 refines these priors toward robust, safety-aware autonomous discovery. 1690

## 5 Discussion & Conclusion 1691

This survey synthesizes recent advances in AI for materials science through a pipeline-centric 1692 lens that connects corpus curation, pre-training, domain adaptation, and instruction tuning to 1693 goal-conditioned, agentic LLMs operating within open-ended experimental environments. By 1694 jointly organizing the field around three guiding questions, it integrates machine-learning and 1695 materials-science perspectives into a single conceptual framework for autonomous materials dis- 1696 covery. The survey systematically reviews predominantly reactive tasks such as prediction, min- 1697 ing, generation, optimization, and verification, together with cross-cutting issues in data, knowl- 1698 edge integration, multimodality, and explainability, clarifying both the strengths and limitations of 1699 current approaches. Building on this foundation, it characterizes emerging agentic systems and 1700 Scientist-AI workflows that couple hypothesis generation, experiment and simulation planning, 1701 and human–AI collaboration, arguing that truly discovery-oriented AI4MatSci requires aligning all 1702 components of the pipeline to real experimental reward signals rather than proxy benchmarks. 1703

Revisiting Fig. 1, the agentic system operating within an open-ended experimental environ- 1704 ment uniquely accesses real-world outcome signals that can be used for credit assignment at the 1705 current decision stage and retroactively propagated to upstream stages spanning pre-training, 1706

materials-specific adaptation, and task design, as detailed in Secs. 2, 3, and 4. From this 1707 vantage point, the concluding section elaborates a coherent roadmap that operationalizes this 1708 pipeline-centric view, integrating the insights from the preceding sections into a unified picture of 1709 how agentic LLM systems can be aligned with end-to-end materials discovery objectives. 1710

#### Environment

Virtual World Real World

[Figure 74]

[Figure 75]

Environment Plasticity

Agent Plasticity

Environment Empowerment

Agent Empowerment

#### Agent

Memory Search

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

Pre-Trained LLM

MatSci-Tuned LLM

Agentic LLM

Figure 8: Search is defined as trial-and-error, generate-and-test, and variation-and-selection, whereas memory is to remember what worked best for each situation and start from there next time. In these processes, certain level of generalization is expected. The pre-trained and MatScituned LLMs serve as the unified memory. The agentic LLM mostly acts as the search operator interacting with the environment, but it can access and store information during interaction into the memory. The agentic LLM can employ the search operator to interact with either the virtual world (e.g., a simulator) or the real world. Following the definition and notations in419, the directed information I(X → Y ) captures the influence that the past elements Xi have on future elements Yi.

Barto420 observed that “reinforcement learning is memorized search” and noted that “in the 1711 beginning, machine learning was reinforcement learning.” From this point of view, general pre- 1712 training, domain-specific fine-tuning and part of lifelong RL in the experimental environment 1713 serve as a unified memory, contributing to the search component of RL, as shown in Fig. 8. 1714 Note that an agentic LLM also relies on the memory function during interacting with the environ- 1715 ment, as it stores useful information into the memory after searching. 1716

However, recent evaluations421 show that frontier LLMs can engage in sophisticated reward 1717 hacking in benchmarking environments, indicating that the memory formed by pre-training and 1718 fine-tuning may overfit scoring artifacts rather than reflect robust performance in real experimen- 1719 tal settings. To mitigate this, experimental environment rewards should be used to reshape up- 1720 stream memory, data curation, objectives, and weights, thus aligning learning with outcome-level 1721 goals. It has also been argued422 that small language models are enough for agentic tasks, as 1722 they are “sufficiently powerful, inherently more suitable, and necessarily more economical for 1723

many invocations in agentic systems”. To some extent, compressing LLMs into small LMs can 1724 be considered as a way to modify the upstream components driven by the credit assignment 1725 process mainly based on the final RL reward – discovering novel materials in our case. Ideally, 1726 the whole pipeline should be tunable, driven by the final objective. 1727

For MatSci scenarios, we can use the influence function423 to track down the contributions of 1728 pre-training and fine-tuning data to the discovery of novel and useful materials. It is also promis- 1729 ing to learn RL rules that are more suitable for MatSci settings, inspired by the success of show- 1730 ing how to discover a state-of-the-art RL rule that outperforms manually designed rules424. We 1731 encourage both the machine learning and materials science communities to view the pipeline as 1732 a whole serving the single ultimate goal – discovering novel materials, using which to scrutinize 1733 the upstream components and redesign them through the lens of RL. 1734

As indicated in Fig. 8, an agentic LLM can operate on either the virtual world and the real 1735 world. It is preferable and practical to first train it on the virtual world and then transfer it and con- 1736 tinually train it in the real-world materials science environment. We should be aware that an agent 1737 cannot learn beyond the knowledge of a simulator, meaning that it is not realistic to expect that 1738 an agent can generalize to the knowledge outside of a simulator designed by human experts and 1739 domain data. This indicates the potential performance gap between an agentic LLM trained in 1740 the virtual world and the one ideally trained in the real-world environment, which is prohibitively 1741 expensive. A closely related and often underemphasized bottleneck is the limited accessibil- 1742 ity of laboratory automation itself. Democratizing lab automation is critical to addressing the 1743 persistent data scarcity challenge in materials science. Despite recent progress, autonomous 1744 and self-driven laboratories remain expensive, slow to deploy, and heavily dependent on expert 1745 knowledge not only in materials science, but also in mechanical engineering, robotics, and soft- 1746 ware engineering. As a result, adapting an existing automated laboratory to a new scientific 1747 objective is still a nontrivial and resource-intensive process, restricting the scale and diversity of 1748 real-world feedback available for learning. 1749

We also envision parallel memory+search across the virtual and real worlds, similar to the 1750 concept of asynchronous thinking proposed by Schilling-Wilhelmi et al.218. 1751

Current benchmarking methods for computational materials discovery mostly focus on fixed 1752 prediction tasks and isolated computational elements, overlooking the inherently iterative, ex- 1753 ploratory, and sometimes serendipitous nature of real scientific discovery425. Even though Crys- 1754 talGym426 provides APIs that are familiar to RL researchers, it still remains unclear if the perfor- 1755 mance can be translated into real-world experimental environments. Inspired by digital twins for 1756 materials373 and co-evolution of agents and the virtual environment427, where a data interface 1757 keeps the digital and physical systems synchronized, we propose to evolve the agent, the virtual 1758 world, and all the components in the upstream stages based on the real-world environmental 1759 reward. While individual components and their associated views each offer limited perspectives, 1760 only a systems-level approach can deliver the complete, integrated understanding required to 1761 objectively accelerate materials innovation; this goal underpins the pipeline-centric view of our 1762 survey. Here, a modular dynamic AI4MatSci testbed is needed, which can be updated with 1763 real-world data more easily. 1764

In terms of additional algorithmic improvement, differently from previous digital-twins for ma- 1765 terials discovery, we argue for a single shared agent or a population of collective agents with 1766 memory428–432 for both simulation and real-world experiments, and even upstream pre- and mid- 1767 training. This unified view is also aligned with the concept depicted in Fig. 8, where the whole 1768 pipeline shares a single memory across all stages, which can be updated by the final real-world 1769 reward signal. 1770

Bengio et al.433 warned about the biosecurity risks at the convergence of AI and the life sci- 1771 ences. Similarly, LLM agent-driven materials discovery and autonomous laboratory workflows 1772 can accelerate beneficial research but also lower barriers for misuse (e.g., designing materials 1773

with hazardous properties or enabling access to dangerous synthesis routes). Thus, it is cru- 1774 cial to develop trustworthy and safe LLM agents for materials science research. We depict the 1775 empowerment-plasticity view419 in Fig. 8. Empowerment is the extent to which an agent can 1776 shape or control the set of future states it can perceive, and plasticity is the capacity of an agent 1777 to maintain adaptability in the face of significant changes or challenges419. Interestingly, the 1778 plasticity of the agent is equivalent to the empowerment of the environment, and vice-versa419. 1779 On the one hand, this implies that, for the safety concerns, the agent might need to dynamically 1780 adjust its plasticity in order to restrain the adversarial empowerment of the environment. On the 1781 other hand, this means that to maximize the overall yields of the whole system over time, the en- 1782 vironment over which the agent is empowered should often have a high plasticity, e.g., with more 1783 controllable components or tools. We argue that AI4MatSci should routinely revisit each module 1784 in the workflow and ask whether it can be brought under learning or optimization pressure rather 1785 than being treated as a frozen heuristic. Concretely, this means not only training neural networks 1786 but also learning data selection and curation policies434, adaptive pre-training corpora, retrieval 1787 and tool-calling strategies, experiment proposal policies, and even aspects of the experimental 1788 protocol itself, using the ultimate reward signals. Our pipeline-centric, end-to-end perspective 1789 thus views the materials discovery loop as a composite system in which as many components 1790 as possible are trainable and where signals from successful or failed discoveries can, over time, 1791 reshape both models and upstream pipeline choices to better serve the ultimate goal of find- 1792 ing novel, useful, and safe materials. Overall, our proposed unified and pipeline-centric view 1793 encourages us to design highly decoupled agents and the corresponding environments to maxi- 1794 mize output while minimizing risks. 1795

## References 1796

- 1. Zhang, W., Wang, Q., Kong, X., Xiong, J., Ni, S., Cao, D., Niu, B., Chen, M., Li, Y., Zhang, 1797 R. et al. (2024). Fine-tuning large language models for chemical text mining. Chemical 1798 science 15, 10600–10611. 1799
- 2. (2024). Structured information extraction from scientific text with large language models. 1800 Nature Communications. 1801
- 3. Van Herck, J., Gil, M.V., Jablonka, K.M., Abrudan, A., Anker, A.S., Asgari, M., Blaiszik, B., 1802 Buffo, A., Choudhury, L., Corminboeuf, C. et al. (2025). Assessment of fine-tuned large 1803 language models for real-world chemistry and material science applications. Chemical 1804 science 16, 670–684. 1805
- 4. Jiang, X., Wang, W., Tian, S., Wang, H., Lookman, T., and Su, Y. (2025). Applications 1806 of natural language processing and large language models in materials discovery. npj 1807 Computational Materials 11, 79. 1808
- 5. Kumbhar, S., Mishra, V., Coutinho, K., Handa, D., Iquebal, A., and Baral, C. (2025). Hy- 1809 pothesis generation for materials discovery and design using goal-driven and constraint- 1810 guided llm agents. arXiv preprint arXiv:2501.13299. 1811
- 6. Liu, G., Sun, M., Matusik, W., Jiang, M., and Chen, J. (2024). Multimodal large lan- 1812 guage models for inverse molecular design with retrosynthetic planning. arXiv preprint 1813 arXiv:2410.04223. 1814

- 7. Kang, C., Liu, X., and Guo, F. (). Retrointext: A multimodal large language model en- 1815 hanced framework for retrosynthetic planning via in-context representation learning. In 1816 The Thirteenth International Conference on Learning Representations. 1817
- 8. Yao, Y., Zhu, J., Liu, Y., Ren, G., Li, X.Y., and Ou, P. (2025). Large language models 1818 for heterogeneous catalysis. Wiley Interdisciplinary Reviews: Computational Molecular 1819 Science 15, e70046. 1820
- 9. Wijk, H., Lin, T., Becker, J., Jawhar, S., Parikh, N., Broadley, T., Chan, L., Chen, M., 1821 Clymer, J., Dhyani, J., Ericheva, E., Garcia, K., Goodrich, B., Jurkovic, N., Karnofsky, H., 1822 Kinniment, M., Lajko, A., Nix, S., Sato, L., Saunders, W., Taran, M., West, B., and Barnes, 1823 E. (2024). RE-Bench: Evaluating frontier AI R&D capabilities of language model agents 1824 against human experts. arXiv:2411.15114. 1825
- 10. Silver, D., Singh, S., Precup, D., and Sutton, R.S. (2021). Reward is enough. Artificial In- 1826 telligence 299, 103535. URL: https://www.sciencedirect.com/science/article/pii/ 1827 S0004370221000862. doi: https://doi.org/10.1016/j.artint.2021.103535. 1828
- 11. Van, M.H., Verma, P., Zhao, C., and Wu, X. (2025). A survey of ai for materials science: 1829 Foundation models, llm agents, datasets, and tools. arXiv:2506.20743. 1830
- 12. Madika, B., Saha, A., Kang, C., Buyantogtokh, B., Agar, J., Wolverton, C.M., Voorhees, P., 1831 Littlewood, P., Kalinin, S., and Hong, S. (2025). Artificial intelligence for materials discov- 1832 ery, development, and optimization. ACS Nano 19, 27116–27158. doi: 10.1021/acsnano. 1833 5c04200. Epub 2025-07-25. 1834
- 13. Pyzer-Knapp, E.O., Manica, M., Staar, P., Morin, L., Ruch, P., Laino, T., Smith, J.R., and 1835 Curioni, A. (2025). Foundation models for materials discovery – current state and future 1836 directions. npj Computational Materials 11, 61. doi: 10.1038/s41524-025-01538-0. 1837
- 14. Peivaste, I., Belouettar, S., Mercuri, F., Fantuzzi, N., Dehghani, H., Izadi, R., Ibrahim, 1838 H., Lengiewicz, J., Belouettar-Mathis, M., Bendine, K., Makradi, A., Horsch, M., Klein, 1839 P., El-Hachemi, M., Preisig, H.A. et al. (2025). Artificial intelligence in materials science 1840 and engineering: Current landscape, key challenges, and future trajectories. Composite 1841 Structures 372, 119419. URL: https://www.sciencedirect.com/science/article/abs/ 1842 pii/S026263822325005847. doi: 10.1016/j.compstruct.2025.119419. Review. 1843
- 15. Jiang, X., Wang, W., and Su, Y. (2025). Applications of natural language process- 1844 ing and large language models in materials discovery. npj Computational Materials 1845 11, 79. URL: https://www.nature.com/articleses/s41524-025-01554-0. doi: 10.1038/ 1846 s41524-025-01554-0. 1847
- 16. Lei, G., Docherty, R., and Cooper, S.J. (2024). Materials science in the era of large lan- 1848 guage models: a perspective. Digital Discovery 3, 1257–1272. 1849
- 17. Pyzer-Knapp, E.O., Pitera, J.W., Staar, P.W.J., Takeda, S., Laino, T., Sanders, D.P., Sexton, 1850 J., Smith, J.R., and Curioni, A. (2022). Accelerating materials discovery using artificial 1851 intelligence, high performance computing and robotics. npj Computational Materials 8, 84. 1852 doi: 10.1038/s41524-022-00765-z. 1853
- 18. Cortes, C., and Vapnik, V. (1995). Support-vector networks. Machine learning 20, 273– 1854

297. 1855

- 19. Breiman, L. (2001). Random forests. Machine learning 45, 5–32. 1856

- 20. Anderson, J.A. (1995). An introduction to neural networks. MIT press. 1857
- 21. McQueen, J.B. (1967). Some methods of classification and analysis of multivariate obser- 1858 vations. In Proc. of 5th Berkeley Symposium on Math. Stat. and Prob. pp. 281–297. 1859
- 22. Khan, K., Rehman, S.U., Aziz, K., Fong, S., and Sarasvady, S. (2014). Dbscan: Past, 1860 present and future. In The fifth international conference on the applications of digital infor- 1861 mation and web technologies (ICADIWT 2014). IEEE pp. 232–238. 1862
- 23. Abdi, H., and Williams, L.J. (2010). Principal component analysis. Wiley interdisciplinary 1863 reviews: computational statistics 2, 433–459. 1864
- 24. Maaten, L.v.d., and Hinton, G. (2008). Visualizing data using t-sne. Journal of machine 1865 learning research 9, 2579–2605. 1866
- 25. Sutton, R.S., Barto, A.G. et al. (1998). Introduction to reinforcement learning vol. 135. MIT 1867 press Cambridge. 1868
- 26. Mnih, V., Kavukcuoglu, K., Silver, D., Graves, A., Antonoglou, I., Wierstra, D., and 1869 Riedmiller, M. (2013). Playing atari with deep reinforcement learning. arXiv preprint 1870 arXiv:1312.5602. 1871
- 27. Schulman, J., Wolski, F., Dhariwal, P., Radford, A., and Klimov, O. (2017). Proximal policy 1872 optimization algorithms. arXiv preprint arXiv:1707.06347. 1873
- 28. Garcia, C.E., Prett, D.M., and Morari, M. (1989). Model predictive control: Theory and 1874 practice—a survey. Automatica 25, 335–348. 1875
- 29. Konda, V., and Tsitsiklis, J. (1999). Actor-critic algorithms. Advances in neural information 1876 processing systems 12. 1877
- 30. Krizhevsky, A., Sutskever, I., and Hinton, G. (2012). Imagenet classification with deep 1878 convolutional neural networks. Neural Information Processing Systems 25. doi: 10.1145/ 1879

3065386. 1880

- 31. Goodfellow, I.J., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., 1881 Courville, A., and Bengio, Y. (2014). Generative adversarial networks. . URL: https: 1882 //arxiv.org/abs/1406.2661. arXiv:1406.2661. 1883
- 32. Kingma, D.P., and Welling, M. (2013). Auto-encoding variational bayes. arXiv preprint 1884 arXiv:1312.6114. 1885
- 33. Computation, N. (2016). Long short-term memory. Neural Comput 9, 1735–1780. 1886
- 34. Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., and Salakhutdinov, R. (2014). 1887 Dropout: a simple way to prevent neural networks from overfitting. The journal of machine 1888 learning research 15, 1929–1958. 1889
- 35. Ioffe, S., and Szegedy, C. (2015). Batch normalization: Accelerating deep network training 1890 by reducing internal covariate shift. In International conference on machine learning. pmlr 1891 pp. 448–456. 1892
- 36. Abadi, M., Agarwal, A., Barham, P., Brevdo, E., Chen, Z., Citro, C., Corrado, G.S., Davis, 1893 A., Dean, J., Devin, M. et al. (2016). Tensorflow: Large-scale machine learning on hetero- 1894 geneous distributed systems. arXiv preprint arXiv:1603.04467. 1895

- 37. Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T., Lin, Z., 1896 Gimelshein, N., Antiga, L. et al. (2019). Pytorch: An imperative style, high-performance 1897 deep learning library. Advances in neural information processing systems 32. 1898
- 38. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Łukasz 1899 Kaiser, and Polosukhin, I. (2017). Attention is all you need. In Advances in Neural In- 1900 formation Processing Systems 30 (NIPS 2017). pp. 5998–6008. 1901
- 39. Devlin, J., Chang, M.W., Lee, K., and Toutanova, K. (2019). BERT: Pre-training of deep 1902 bidirectional transformers for language understanding. In Proceedings of the 2019 Confer- 1903 ence of the North American Chapter of the Association for Computational Linguistics: Hu- 1904 man Language Technologies, Volume 1 (Long and Short Papers). Minneapolis, Minnesota: 1905 Association for Computational Linguistics pp. 4171–4186. doi: 10.18653/v1/N19-1423. 1906
- 40. Brown, T.B., Mann, B., Ryder, N., Subbiah, M., Kaplan, J., Dhariwal, P., Neelakan- 1907 tan, A., Shyam, P., Saxe, A., Askell, A., Agarwal, S., Herbert-Voss, A., Krueger, 1908 G., Henighan, T., Child, R., Ramesh, A., Ziegler, D.M., Wu, J., Winter, C., Hesse, 1909 C., Chen, M., Sigler, E., Litwin, M., Gray, S., Chess, B., Clark, J., Berner, C., Mc- 1910 Candlish, S., Radford, A., Sutskever, I., and Amodei, D. (2020). Language mod- 1911 els are few-shot learners. In Advances in Neural Information Processing Systems 1912 vol. 33. pp. 1877–1901. URL: https://proceedings.neurips.cc/paper/2020/hash/ 1913 1457c0d6bfcb4967418bfb8ac142f64a-Abstract.html. 1914
- 41. Bommasani, R., Hudson, D.A., Adeli, E., Altman, R., Arora, S., von Arx, S., Bernstein, 1915 M.S., Bohg, J., Bosselut, A., Brunskill, E., Byrd, D., ..., and Liang, P. (2021). On the 1916 opportunities and risks of foundation models. arXiv preprint arXiv:2108.07258. URL: 1917 https://arxiv.org/abs/2108.07258. arXiv:2108.07258. 1918
- 42. Beltagy, I., Lo, K., and Cohan, A. (2019). Scibert: A pretrained language model for sci- 1919 entific text. In Proceedings of the 2019 Conference on Empirical Methods in Natural 1920 Language Processing and the 9th International Joint Conference on Natural Language 1921 Processing (EMNLP-IJCNLP). Hong Kong, China: Association for Computational Lin- 1922 guistics pp. 3615–3620. URL: https://aclanthology.org/D19-1371. doi: 10.18653/v1/ 1923 D19-1371. 1924
- 43. Jumper, J., Evans, R., Pritzel, A., Green, T., Figurnov, M., Ronneberger, O., Tunyasuvu- 1925 nakool, K., Bates, R., Z´ˇıdek, A., Potapenko, A., Bridgland, A., Meyer, C., Kohl, S.A.A., 1926 Ballard, A.J., Cowie, A., Romera-Paredes, B., Nikolov, S., Jain, R., Adler, J., Back, T., Pe- 1927 tersen, S., Reiman, D., Clancy, E., Zielinski, M., Steinegger, M., Pacholska, M., Bergham- 1928 mer, T., Bodenstein, S., Silver, D., Vinyals, O., Senior, A.W., Kavukcuoglu, K., Kohli, P., and 1929 Hassabis, D. (2021). Highly accurate protein structure prediction with AlphaFold. Nature 1930 596, 583–589. doi: 10.1038/s41586-021-03819-2. 1931
- 44. Wei, J., Wang, X., Schuurmans, D., Bosma, M., ichter, b., Xia, F., Chi, E., Le, Q.V., and 1932 Zhou, D. (2022). Chain-of-thought prompting elicits reasoning in large language mod- 1933 els. In S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh, 1934 eds. Advances in Neural Information Processing Systems vol. 35. Curran Associates, Inc. 1935 pp. 24824–24837. URL: https://proceedings.neurips.cc/paper_files/paper/2022/ 1936 file/9d5609613524ecf4f15af0f7b31abca4-Paper-Conference.pdf. 1937
- 45. Borgeaud, S., Mensch, A., Hoffmann, J., Cai, T., Rutherford, E., Millican, K., Van 1938 Den Driessche, G.B., Lespiau, J.B., Damoc, B., Clark, A., De Las Casas, D., Guy, A., 1939

- Menick, J., Ring, R., Hennigan, T., Huang, S., Maggiore, L., Jones, C., Cassirer, A., 1940 Brock, A., Paganini, M., Irving, G., Vinyals, O., Osindero, S., Simonyan, K., Rae, J., 1941 Elsen, E., and Sifre, L. (2022). Improving language models by retrieving from trillions 1942 of tokens. In K. Chaudhuri, S. Jegelka, L. Song, C. Szepesvari, G. Niu, and S. 1943 Sabato, eds. Proceedings of the 39th International Conference on Machine Learning 1944 vol. 162 of Proceedings of Machine Learning Research. PMLR pp. 2206–2240. URL: 1945 https://proceedings.mlr.press/v162/borgeaud22a.html. 1946
- 46. Kaplan, J., McCandlish, S., Henighan, T.J., Brown, T.B., Chess, B., Child, R., Gray, S., 1947 Radford, A., Wu, J., and Amodei, D. (2020). Scaling laws for neural language models. 1948 ArXiv abs/2001.08361. URL: https://api.semanticscholar.org/CorpusID:210861095. 1949
- 47. Yu, S., Ran, N., and Liu, J. (2024). Large-language models: The game-changers for ma- 1950 terials science research. Artificial Intelligence Chemistry 2, 100076. 1951
- 48. Wei, L., Li, Q., Song, Y., Stefanov, S., Dong, R., Fu, N., Siriwardane, E.M., Chen, F., and 1952 Hu, J. (2024). Crystal composition transformer: Self-learning neural language model for 1953 generative and tinkering design of materials. Advanced Science 11, 2304305. 1954
- 49. Kim, H., Na, J., and Lee, W.B. (2021). Generative chemical transformer: neural machine 1955 learning of molecular geometric structures from chemical language via attention. Journal 1956 of chemical information and modeling 61, 5804–5814. 1957
- 50. Li, J., Tang, J., Zhao, W.X., Nie, J.Y., and Wen, J.R. (2021). M6: Multi-modality to multi- 1958 modality multitask mega-transformer for unified pretraining. In Proceedings of the 27th 1959 ACM SIGKDD Conference on Knowledge Discovery & Data Mining. ACM pp. 3251–3261. 1960 doi: 10.1145/3447548.3467206. 1961
- 51. Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, 1962 A., Mishkin, P., Clark, J. et al. (2021). Learning transferable visual models from natural 1963 language supervision. In International conference on machine learning. PmLR pp. 8748– 1964

8763. 1965

- 52. Li, J., Li, D., Xiong, C., and Hoi, S.C.H. (2022). BLIP: Bootstrapping language-image pre- 1966 training for unified vision-language understanding and generation. In Proceedings of the 1967 39th International Conference on Machine Learning (ICML) vol. 162. PMLR pp. 12888– 1968

12900. URL: https://proceedings.mlr.press/v162/li22n/li22n.pdf. 1969

- 53. Tang, Y., Xu, W., Cao, J., Gao, W., Farrell, S., Erichson, B., Mahoney, M.W., Nonaka, 1970 A., and Yao, Z. (2025). Matterchat: A multi-modal llm for material science. arXiv preprint 1971 arXiv:2502.13107. 1972
- 54. Choudhary, K. (2025). Microscopygpt: Generating atomic-structure captions from mi- 1973 croscopy images of 2d materials with vision-language transformers. The Journal of Phys- 1974 ical Chemistry Letters 16, 7028–7035. 1975
- 55. Adak, D., Rawat, Y.S., and Vyas, S. (2025). Molvision: Molecular property prediction with 1976 vision language models. arXiv preprint arXiv:2507.03283. 1977
- 56. Gu, A., Goel, K., and Re,´ C. (2021). Efficiently modeling long sequences with struc- 1978 tured state spaces. arXiv preprint arXiv:2111.00396. URL: https://arxiv.org/abs/2111. 1979

###### 00396. doi: 10.48550/arXiv.2111.00396. 1980

- 57. Gu, A., and Dao, T. (2024). Mamba: Linear-Time Sequence Modeling with Selective State 1981 Spaces. arXiv preprint arXiv:2312.00752. URL: https://arxiv.org/abs/2312.00752. doi: 1982 10.48550/arXiv.2312.00752. 1983
- 58. Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C., Mishkin, P., Zhang, C., Agarwal, 1984 S., Slama, K., Ray, A. et al. (2022). Training language models to follow instructions with 1985 human feedback. Advances in neural information processing systems 35, 27730–27744. 1986
- 59. Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K., and Cao, Y. (2023). ReAct: 1987 Synergizing reasoning and acting in language models. In International Conference on 1988 Learning Representations. 1989
- 60. Schick, T., Dwivedi-Yu, J., Dess`ı, R., Raileanu, R., Lomeli, M., Zettlemoyer, L., Cancedda, 1990 N., and Scialom, T. (2023). Toolformer: Language models can teach themselves to use 1991 tools. arXiv preprint arXiv:2302.04761. URL: https://arxiv.org/abs/2302.04761. 1992
- 61. Gravitas, S. (2023). Autogpt: Build, deploy, and run ai agents. . URL: https://github. 1993 com/Significant-Gravitas/AutoGPT latest release: October 22, 2025. 1994
- 62. Anthropic (2025). Claude 3.7 sonnet and claude code. Anthropic News. . URL: https: 1995 //www.anthropic.com/news/claude-3-7-sonnet accessed November 2025. 1996
- 63. Shaw, F.X. (2025). Microsoft build 2025: The age of ai agents and building the open agen- 1997 tic web. Blog post, Microsoft. . URL: https://blogs.microsoft.com/blog/2025/05/19/ 1998 microsoft-build-2025-the-age-of-ai-agents-and-building-the-open-agentic-web/ 1999 accessed [insert date you accessed]. 2000
- 64. Jiang, Q., and Karniadakis, G. (2025). Agenticsciml: Collaborative multi-agent systems for 2001 emergent discovery in scientific machine learning. . URL: https://arxiv.org/abs/2511. 2002

07262. arXiv:2511.07262. 2003

- 65. Ward, L., Agrawal, A., Choudhary, A., and Wolverton, C. (2016). A general-purpose ma- 2004 chine learning framework for predicting properties of inorganic materials. npj Computa- 2005 tional Materials 2, 16028. doi: 10.1038/npjcompumats.2016.28. 2006
- 66. Ximendes, E., Marin, R., Carlos, L.D. et al. (2022). Less is more: dimensionality re- 2007 duction as a general strategy for more precise luminescence thermometry. Light: Sci- 2008 ence & Applications 11, 237. URL: https://doi.org/10.1038/s41377-022-00932-3. doi: 2009 10.1038/s41377-022-00932-3. 2010
- 67. Kim, H., Choi, H., Kang, D., Lee, W.B., and Na, J. (2024). Materials discovery with ex- 2011 treme properties via reinforcement learning-guided combinatorial chemistry. Chemical 2012 Science 15, 7908–7925. URL: http://dx.doi.org/10.1039/D3SC05281H. doi: 10.1039/ 2013 D3SC05281H. 2014
- 68. Choudhary, K., DeCost, B., Chen, C. et al. (2022). Recent advances and applications of 2015 deep learning methods in materials science. npj Computational Materials 8. URL: https: 2016 //doi.org/10.1038/s41524-022-00734-6. doi: 10.1038/s41524-022-00734-6. 2017
- 69. LeCun, Y., Bengio, Y., and Hinton, G. (2015). Deep learning. Nature 521, 436–444. doi: 2018 10.1038/nature14539. 2019
- 70. Goodfellow, I., Bengio, Y., and Courville, A. (2016). Deep Learning. MIT Press. http: 2020 //www.deeplearningbook.org. 2021

- 71. Xie, T., and Grossman, J.C. (2018). Crystal graph convolutional neural networks for 2022 an accurate and interpretable prediction of material properties. Phys. Rev. Lett. 120, 2023

145301. URL: https://link.aps.org/doi/10.1103/PhysRevLett.120.145301. doi: 10. 2024 1103/PhysRevLett.120.145301. 2025

- 72. Jha, D., Ward, L., Paul, A., Liao, W.k., Choudhary, A., Wolverton, C., and Agrawal, A. 2026

(2018). ElemNet: deep learning the chemistry of materials from only elemental composi- 2027 tion. Scientific reports 8, 17593. doi: 10.1038/s41598-018-35934-y. 2028

- 73. Butler, K.T., Davies, D.W., Cartwright, H., Isayev, O., and Walsh, A. (2018). Machine 2029 learning for molecular and materials science. Nature 559, 547–555. doi: 10.1038/ 2030 s41586-018-0337-2. 2031
- 74. Cho, K., Van Merrienboer,¨ B., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk, H., 2032 and Bengio, Y. (2014). Learning phrase representations using rnn encoder-decoder for 2033 statistical machine translation. arXiv preprint arXiv:1406.1078. 2034
- 75. Kailkhura, B., Gallagher, B., Kim, S., Hiszpanski, A., and Han, T.Y.J. (2019). Reliable and 2035 explainable machine-learning methods for accelerated material discovery. npj Computa- 2036 tional Materials 5, 108. 2037
- 76. Weininger, D. (1988). Smiles, a chemical language and information system. 1. introduction 2038 to methodology and encoding rules. J. Chem. Inf. Comput. Sci. 28, 31–36. URL: https: 2039 //doi.org/10.1021/ci00057a005. doi: 10.1021/ci00057a005. 2040
- 77. Chen, M., Tworek, J., Jun, H., Yuan, Q., Ponde,´ H., Kaplan, J., Edwards, H., Burda, Y., 2041 Joseph, N., Brockman, G., Ray, A., Puri, R., Krueger, G., Petrov, M., Khlaaf, H., Sastry, 2042 G., Mishkin, P., Chan, B., Gray, S., Ryder, N., Pavlov, M., Power, A., Kaiser, L., Bavarian, 2043 M., Winter, C., Tillet, P., Such, F.P., Cummings, D.W., Plappert, M., Chantzis, F., Barnes, 2044 E., Herbert-Voss, A., Guss, W.H., Nichol, A., Babuschkin, I., Balaji, S., Jain, S., Carr, A., 2045 Leike, J., Achiam, J., Misra, V., Morikawa, E., Radford, A., Knight, M.M., Brundage, M., 2046 Murati, M., Mayer, K., Welinder, P., McGrew, B., Amodei, D., McCandlish, S., Sutskever, 2047

I., and Zaremba, W. (2021). Evaluating large language models trained on code. ArXiv 2048 abs/2107.03374. URL: https://api.semanticscholar.org/CorpusID:235755472. 2049

- 78. Nijkamp, E., Pang, B., Hayashi, H., Tu, L., Wang, H., Zhou, Y., Savarese, S., and Xiong, 2050 C. (2022). Codegen: An open large language model for code with multi-turn program 2051 synthesis. In International Conference on Learning Representations. URL: https://api. 2052 semanticscholar.org/CorpusID:252668917. 2053
- 79. Lewkowycz, A., Andreassen, A., Dohan, D., Dyer, E., Michalewski, H., Ramasesh, V., 2054 Slone, A., Anil, C., Schlag, I., Gutman-Solo, T., Wu, Y., Neyshabur, B., Gur-Ari, G., and 2055 Misra, V. (2022). Solving quantitative reasoning problems with language models. In Pro- 2056 ceedings of the 36th International Conference on Neural Information Processing Systems. 2057 NIPS ’22. Red Hook, NY, USA: Curran Associates Inc. ISBN 9781713871088. 2058
- 80. Cobbe, K., Kosaraju, V., Bavarian, M., Chen, M., Jun, H., Kaiser, L., Plappert, M., Tworek, 2059 J., Hilton, J., Nakano, R., Hesse, C., and Schulman, J. (2021). Training verifiers to solve 2060 math word problems. ArXiv abs/2110.14168. URL: https://api.semanticscholar.org/ 2061 CorpusID:239998651. 2062
- 81. Chalkidis, I., Fergadiotis, M., Malakasiotis, P., Aletras, N., and Androutsopoulos, I. (2020). 2063 LEGAL-BERT: The muppets straight out of law school. In T. Cohn, Y. He, and Y. Liu, eds. 2064

- Findings of the Association for Computational Linguistics: EMNLP 2020. Online: Associ- 2065 ation for Computational Linguistics pp. 2898–2904. URL: https://aclanthology.org/ 2066 2020.findings-emnlp.261/. doi: 10.18653/v1/2020.findings-emnlp.261. 2067
- 82. Ji, Z., Lee, N., Frieske, R., Yu, T., Su, D., Xu, Y., Ishii, E., Bang, Y.J., Madotto, A., and 2068 Fung, P. (2023). Survey of hallucination in natural language generation. ACM Comput. 2069 Surv. 55. URL: https://doi.org/10.1145/3571730. doi: 10.1145/3571730. 2070
- 83. Li, W., Wu, W., Chen, M., Liu, J., Xiao, X., and Wu, H. (2022). Faithfulness in natural lan- 2071 guage generation: A systematic survey of analysis, evaluation and optimization methods. 2072 ArXiv abs/2203.05227. URL: https://api.semanticscholar.org/CorpusID:247362526. 2073
- 84. Dziri, N., Lu, X., Sclar, M., Li, X.L., Jiang, L., Lin, B.Y., West, P., Bhagavatula, C., Bras, 2074 R.L., Hwang, J.D., Sanyal, S., Welleck, S., Ren, X., Ettinger, A., Harchaoui, Z., and Choi, 2075 Y. (2023). Faith and fate: Limits of transformers on compositionality. . URL: https:// 2076 arxiv.org/abs/2305.18654. arXiv:2305.18654. 2077
- 85. Chanussot, L., Das, A., Goyal, S., Lavril, T., Shuaibi, M., Riviere, M., Tran, K., Heras- 2078 Domingo, J., Ho, C., Hu, W., Palizhati, A., Sriram, A., Wood, B., Yoon, J., Parikh, D., 2079 Zitnick, C.L., and Ulissi, Z. (2021). Open catalyst 2020 (oc20) dataset and community 2080 challenges. ACS Catalysis 11, 6059–6072. URL: http://dx.doi.org/10.1021/acscatal. 2081 0c04525. doi: 10.1021/acscatal.0c04525. 2082
- 86. Jain, A. et al. (2013). The materials project: A materials genome approach to accelerating 2083 materials innovation. APL Materials 1, 011002. doi: 10.1063/1.4812323. 2084
- 87. Ramakrishnan, R., Dral, P.O., Dral, P.O., Rupp, M., and von Lilienfeld, O.A. (2014). Quan- 2085 tum chemistry structures and properties of 134 kilo molecules. Scientific Data 1. URL: 2086 https://api.semanticscholar.org/CorpusID:15367821. 2087
- 88. Singh, A., Hu, R., Goswami, V., Couairon, G., Galuba, W., Rohrbach, M., and Kiela, D. 2088

(2022). FLAVA: A foundational language and vision alignment model. In Proceedings 2089 of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 2090 15617–15629. doi: 10.1109/CVPR52688.2022.01519. 2091

- 89. Alayrac, J.B., Donahue, J., Luc, P., Miech, A., Barr, I., Hasson, Y., Lenc, K., Mensch, A., 2092 Millican, K., Reynolds, M., Ring, R., Rutherford, E., Cabi, S., Han, T., Gong, Z., Saman- 2093 gooei, S., Monteiro, M., Menick, J., Borgeaud, S., Brock, A., Nematzadeh, A., Sharifzadeh, 2094 S., Binkowski, M., Barreira, R., Vinyals, O., Zisserman, K., and Simonyan, K. (2022). 2095 Flamingo: a visual language model for few-shot learning. In Advances in Neural Informa- 2096 tion Processing Systems 35 (NeurIPS 2022). URL: https://arxiv.org/abs/2204.14198. 2097
- 90. Peng, Z., Wang, W., Dong, L., Hao, Y., Huang, S., Ma, S., and Wei, F. (2023). 2098 Kosmos-2: Grounding multimodal large language models to the world. arXiv preprint 2099 arXiv:2306.14824. URL: https://arxiv.org/abs/2306.14824. 2100
- 91. Lin, J., Yin, H., Ping, W., Lu, Y., Molchanov, P., Tao, A., Mao, H., Kautz, J., Shoeybi, M., and 2101 Han, S. (2024). Vila: On pre-training for visual language models. In Proceedings of CVPR 2102

2024. URL: https://openaccess.thecvf.com/content/CVPR2024/papers/Lin_VILA_On_ 2103 Pre-training_for_Visual_Language_Models_CVPR_2024_paper.pdf. 2104

- 92. Zhang, T., and Yang, D.B. (2025). Multimodal machine learning with large language em- 2105 bedding model for polymer property prediction. Chemistry of Materials 37, 7002–7013. 2106

- 93. Silver, D., and Sutton, R.S. (2025). Welcome to the era of experience. Preprint. 2107 URL: https://storage.googleapis.com/deepmind-media/Era-of-Experience%20/The% 2108 20Era%20of%20Experience%20Paper.pdf. 2109
- 94. Wang, H., Fu, T., Du, Y., Gao, W., Huang, K., Liu, Z., Chandak, P., Liu, S., Van Katwyk, 2110 P., Deac, A., Anandkumar, A., Bergen, K., Gomes, C.P., Ho, S., Kohli, P., Lasenby, J., 2111 Leskovec, J., Liu, T.Y., Manrai, A., Marks, D., Ramsundar, B., Song, L., Sun, J., Tang, 2112 J., Velickoviˇ c,´ P., Welling, M., Zhang, L., Coley, C.W., Bengio, Y., and Zitnik, M. (2023). 2113 Scientific discovery in the age of artificial intelligence. Nature 620, 47–60. URL: https: 2114 //doi.org/10.1038/s41586-023-06221-2. doi: 10.1038/s41586-023-06221-2. 2115
- 95. Bai, Y., Kadavath, S., Kundu, S., Askell, A., Kernion, J., Jones, A., Chen, A., Goldie, 2116 A., Mirhoseini, A., McKinnon, C. et al. (2022). Constitutional ai: Harmlessness from ai 2117 feedback. arXiv preprint arXiv:2212.08073. 2118
- 96. Finn, C., Abbeel, P., and Levine, S. (2017). Model-agnostic meta-learning for fast adapta- 2119 tion of deep networks. In International conference on machine learning. PMLR pp. 1126– 2120

1135. 2121

- 97. Garg, S., Tsipras, D., Liang, P., and Valiant, G. (2022). What can transformers learn in- 2122 context? a case study of simple function classes. In Proceedings of the 36th International 2123 Conference on Neural Information Processing Systems. NIPS ’22. Red Hook, NY, USA: 2124 Curran Associates Inc. ISBN 9781713871088. 2125
- 98. Chen, Z., Xie, Y., Wu, Y., Lin, Y., Tomiya, S., and Lin, J. (2024). An interpretable and 2126 transferrable vision transformer model for rapid materials spectra classification. Digital 2127 Discovery 3, 369–380. 2128
- 99. Xie, T., and Grossman, J.C. (2018). Crystal graph convolutional neural networks for 2129 an accurate and interpretable prediction of material properties. Phys. Rev. Lett. 120, 2130

145301. URL: https://link.aps.org/doi/10.1103/PhysRevLett.120.145301. doi: 10. 2131 1103/PhysRevLett.120.145301. 2132

- 100. Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., K¨uttler, H., Lewis, M., 2133 Yih, W.t., Rocktaschel,¨ T., Riedel, S., and Kiela, D. (2020). Retrieval-augmented genera- 2134 tion for knowledge-intensive nlp tasks. In Proceedings of the 34th International Conference 2135 on Neural Information Processing Systems. NIPS ’20. Red Hook, NY, USA: Curran Asso- 2136 ciates Inc. ISBN 9781713829546. 2137
- 101. Gao, L., Madaan, A., Zhou, S., Alon, U., Liu, P., Yang, Y., Callan, J., and Neubig, G. (2023). 2138 Pal: program-aided language models. In Proceedings of the 40th International Conference 2139 on Machine Learning. ICML’23. JMLR.org. 2140
- 102. Nakano, R., Hilton, J., Balaji, S., Wu, J., Long, O., Kim, C., Hesse, C., Jain, S., Kosaraju, 2141 V., Saunders, W., Jiang, X., Cobbe, K., Eloundou, T., Krueger, G., Button, K., Knight, 2142 M., Chess, B., and Schulman, J. (2021). Webgpt: Browser-assisted question-answering 2143 with human feedback. ArXiv abs/2112.09332. URL: https://api.semanticscholar.org/ 2144 CorpusID:245329531. 2145
- 103. Qin, Y., Liang, S., Ye, Y., Zhu, K., Yan, L., Lu, Y., Lin, Y., Cong, X., Tang, X., Qian, B., Zhao, 2146 S., Tian, R., Xie, R., Zhou, J., Gerstein, M., Li, D., Liu, Z., and Sun, M. (2023). Toolllm: 2147 Facilitating large language models to master 16000+ real-world apis. . arXiv:2307.16789. 2148

- 104. Chase, H. (2022). LangChain. . URL: https://github.com/langchain-ai/langchain. 2149
- 105. Gemini Robotics Team, Abdolmaleki, A., Abeyruwan, S., Ainslie, J., Alayrac, J.B., Arenas, 2150 M.G., Balakrishna, A., Batchelor, N., Bewley, A., Bingham, J., Bloesch, M., Bousmalis, K., 2151 Brakel, P., Brohan, A., Buschmann, T., Byravan, A., Cabi, S., Caluwaerts, K., Casarini, 2152 F., Chan, C., Chang, O., Chappellet-Volpini, L., Chen, J.E., Chen, X., Chiang, H.T.L., 2153 Choromanski, K., Collister, A., D’Ambrosio, D.B., Dasari, S., Davchev, T., Dave, M.K., 2154 Devin, C., Di Palo, N., Ding, T., Doersch, C., Dostmohamed, A., Du, Y., Dwibedi, D., 2155 Egambaram, S.T., Elabd, M., Erez, T., Fang, X., Fantacci, C., Fong, C., Frey, E., Fu, C., 2156 Gao, R., Giustina, M., Gopalakrishnan, K., Graesser, L., Groth, O., Gupta, A., Hafner, 2157 R., Hansen, S., Hasenclever, L., Haves, S., Heess, N., Hernaez, B., Hofer, A., Hsu, J., 2158 Huang, L., Huang, S.H., Iscen, A., Jacob, M.G., Jain, D., Jesmonth, S., Jindal, A., Julian, 2159 R., Kalashnikov, D., Karagozler, M.E., Karp, S., Kecman, M., Kew, J.C., Kim, D., Kim, F., 2160 Kim, J., Kipf, T., Kirmani, S., Konyushkova, K., Ku, L.Y., Kuang, Y., Lampe, T., Laurens, A., 2161 Le, T.A., Leal, I., Lee, A.X., Lee, T.W.E., Lever, G., Liang, J., Lin, L.H., Liu, F., Long, S., Lu, 2162 C., Maddineni, S., Majumdar, A., Maninis, K.K., Marmon, A., Martinez, S., Michaely, A.H. 2163 et al. (2025). Gemini robotics 1.5: Pushing the frontier of generalist robots with advanced 2164 embodied reasoning, thinking, and motion transfer. arXiv:2510.03342. 2165
- 106. NVIDIA, Azzolini, A., Brandon, H., Chattopadhyay, P., Chen, H., Chu, J., Cui, Y., Diamond, 2166 J., Ding, Y., Ferroni, F., Govindaraju, R., Gu, J., Gururani, S., El Hanafi, I., Hao, Z., Huff- 2167 man, J., Jin, J., Johnson, B., Khan, R., Kurian, G., Lantz, E., Lee, N., Li, Z., Li, X., Lin, T.Y., 2168 Lin, Y.C., Liu, M.Y., Mathau, A., Ni, Y., Pavao, L., Ping, W., Romero, D.W., Smelyanskiy, 2169 M., Song, S., Tchapmi, L., Wang, A.Z., Wang, B., Wang, H., Wei, F., Xu, J., Xu, Y., Yang, 2170 X., Yang, Z., Zeng, X., and Zhang, Z. (2025). Cosmos-reason1: From physical common 2171 sense to embodied reasoning. . URL: https://arxiv.org/abs/2503.15558. 2172
- 107. M. Bran, A., Cox, S., Schilter, O., Baldassari, C., White, A.D., and Schwaller, P. 2173

(2024). Augmenting large language models with chemistry tools. Nature Machine Intelli- 2174 gence 6, 525–535. URL: https://doi.org/10.1038/s42256-024-00832-8. doi: 10.1038/ 2175 s42256-024-00832-8. 2176

- 108. Lu, C., Lu, C., Lange, R.T., Foerster, J., Clune, J., and Ha, D. (2024). The ai scientist: 2177 Towards fully automated open-ended scientific discovery. . URL: https://arxiv.org/ 2178 abs/2408.06292. arXiv:2408.06292. 2179
- 109. Jolicoeur-Martineau, A. (2025). Less is more: Recursive reasoning with tiny networks. 2180 arXiv preprint arXiv:2510.04871. 2181
- 110. Souly, A., Rando, J., Chapman, E., Davies, X., Hasircioglu, B., Shereen, E., Mougan, 2182 C., Mavroudis, V., Jones, E., Hicks, C. et al. (2025). Poisoning attacks on llms require a 2183 near-constant number of poison samples. arXiv preprint arXiv:2510.07192. 2184
- 111. Li, Z., Zhao, W., Li, Y., and Sun, J. (2024). Do influence functions work on large language 2185 models? arXiv:2409.19998. 2186
- 112. Grosse, R., Bae, J., Anil, C., Elhage, N., Tamkin, A., Tajdini, A., Steiner, B., Li, D., Durmus, 2187 E., Perez, E., Hubinger, E., Lukosi¯ˇ ute,˙ K., Nguyen, K., Joseph, N., McCandlish, S., Kaplan, 2188 J., and Bowman, S.R. (2023). Studying large language model generalization with influence 2189 functions. arXiv:2308.03296. 2190

- 113. Jin, L., Du, Z., Shu, L., Mei, Y., and Zhang, H. (2024). Crystal transformer based universal 2191 atomic embedding for accurate and transferable prediction of materials properties. arXiv 2192 preprint arXiv:2401.09755. 2193
- 114. Jin, L., Du, Z., Shu, L., Cen, Y., Xu, Y., Mei, Y., and Zhang, H. (2025). Transformer- 2194 generated atomic embeddings to enhance prediction accuracy of crystal properties with 2195 machine learning. Nature Communications 16, 1210. 2196
- 115. Li, Y., Han, K., Zhang, Z., Chen, X., Wang, Y., Rong, Y., Gonzalez, J.E., and You, Y. (2023). 2197 Materials informatics transformer: A language model for interpretable materials properties 2198 prediction. arXiv preprint arXiv:2308.16259. 2199
- 116. Lee, J., Kim, D.K., Kim, S.W., Lee, S.B., and Kim, H.S. (2022). Evaluation of principal 2200 features for predicting bulk and shear modulus of inorganic solids with machine learning. 2201 Scientific Reports 12, 1–11. 2202
- 117. Liu, S., Wen, T., Ye, B., Li, Z., and Srolovitz, D.J. (2024). Large language models for ma- 2203 terial property predictions: elastic constant tensor prediction and materials design. arXiv 2204 preprint arXiv:2405.11975. 2205
- 118. Faber, F., Lindmaa, A., Von Lilienfeld, O.A., and Armiento, R. (2015). Crystal structure 2206 representations for machine learning models of formation energies. International Journal 2207 of Quantum Chemistry 115, 1094–1101. 2208
- 119. Houchins, G., Yuan, Z., and Arroyave, R. (2024). Formation energy prediction of material 2209 crystal structures using deep learning. arXiv preprint arXiv:2401.00859. 2210
- 120. Ghosh, S., and Tewari, A. (2025). Automated extraction of material properties using llm- 2211 based ai agents. arXiv preprint arXiv:2510.01235. 2212
- 121. Korop, M.M., and Prybyla, A.V. (2025). Application of llm to search and systematize the 2213 properties of thermoelectric materials in scientific literature. Journal of Thermoelectricity 2214 pp. 16–25. 2215
- 122. Bran, A.M., Cox, S., Schilter, O., Baldassari, C., White, A.D., and Schwaller, P. (2024). 2216 ChemCrow: Augmenting large-language models with chemistry tools. Nature Machine 2217 Intelligence 6, 525–535. doi: 10.1038/s42256-024-00832-8. 2218
- 123. Boiko, D.A., MacKnight, R., and Gomes, G. (2023). Emergent autonomous scientific re- 2219 search capabilities of large language models. arXiv preprint arXiv:2304.05332. 2220
- 124. Du, Z., Jin, L., Shu, L., Cen, Y., Xu, Y., Mei, Y., and Zhang, H. (2024). Ctgnn: Crystal 2221 transformer graph neural network for crystal material property prediction. arXiv preprint 2222 arXiv:2405.11502. 2223
- 125. Li, Y., Gupta, V., Kilic, M.N.T., Choudhary, K., Wines, D., Liao, W.k., Choudhary, A., and 2224 Agrawal, A. (2025). Hybrid-llm-gnn: integrating large language models and graph neural 2225 networks for enhanced materials property prediction. Digital Discovery 4, 376–383. 2226
- 126. Caruana, R. (1997). Multitask learning. Machine learning 28, 41–75. 2227
- 127. Prein, T., Pan, E., Doerr, T., Olivetti, E., and Rupp, J.L. (2023). Mtencoder: A multi-task 2228 pretrained transformer encoder for materials representation learning. In AI for Accelerated 2229 Materials Design-NeurIPS 2023 Workshop. 2230

- 128. Tavazza, F., DeCost, B., and Choudhary, K. (2021). Uncertainty prediction for machine 2231 learning models of material properties. ACS omega 6, 32431–32440. 2232
- 129. Kim, E., Huang, K., Tomala, A., Matthews, S., Strubell, E., Saunders, A., McCallum, A., 2233 and Olivetti, E. (2017). Machine-learned and codified synthesis parameters of oxide ma- 2234 terials. Scientific data 4, 1–9. 2235
- 130. Huang, S., and Cole, J.M. (2020). A database of battery materials auto-generated using 2236 chemdataextractor. Scientific Data 7, 260. 2237
- 131. Sierepeklis, O., and Cole, J.M. (2022). A thermoelectric materials database auto- 2238 generated from the scientific literature using chemdataextractor. Scientific Data 9, 648. 2239
- 132. Kumar, P., Kabra, S., and Cole, J.M. (2024). a database of stress-strain properties auto- 2240 generated from the scientific literature using chemdataextractor. Scientific Data 11, 1273. 2241
- 133. Wang, W., Jiang, X., Tian, S., Liu, P., Lookman, T., Su, Y., and Xie, J. (2023). Alloy syn- 2242 thesis and processing by semi-supervised text mining. npj Computational Materials 9, 2243

183. 2244

- 134. Court, C.J., and Cole, J.M. (2020). Magnetic and superconducting phase diagrams and 2245 transition temperatures predicted using text mining and machine learning. npj Computa- 2246 tional Materials 6, 18. 2247
- 135. Venugopal, V., and Olivetti, E. (2024). Matkg: An autonomously generated knowledge 2248 graph in material science. Scientific Data 11, 217. 2249
- 136. Zhang, Y., Chen, F., Liu, Z., Ju, Y., Cui, D., Zhu, J., Jiang, X., Guo, X., He, J., Zhang, 2250 L. et al. (2024). A materials terminology knowledge graph automatically constructed from 2251 text corpus. Scientific Data 11, 600. 2252
- 137. Yan, R., Jiang, X., Wang, W., Dang, D., and Su, Y. (2022). Materials information extraction 2253 via automatically generated corpus. Scientific Data 9, 401. 2254
- 138. Zhang, L., and Stricker, M. (2023). Matnexus: A comprehensive text mining and analysis 2255 suite for materials discover. arXiv preprint arXiv:2311.06303. 2256
- 139. Guha, S., Mullick, A., Agrawal, J., Ram, S., Ghui, S., Lee, S.C., Bhattacharjee, S., and 2257 Goyal, P. (2021). Matscie: An automated tool for the generation of databases of methods 2258 and parameters used in the computational materials science literature. Computational 2259 Materials Science 192, 110325. 2260
- 140. Antunes, L.M., Butler, K.T., and Grau-Crespo, R. (2024). Crystal structure generation with 2261 autoregressive large language modeling. . URL: https://arxiv.org/abs/2307.04340. 2262 arXiv:2307.04340. 2263
- 141. Gruver, N., Sriram, A., Madotto, A., Wilson, A.G., Zitnick, C.L., and Ulissi, Z.W. (2024). 2264 Fine-tuned language models generate stable inorganic materials as text. In The Twelfth 2265 International Conference on Learning Representations. URL: https://openreview.net/ 2266 forum?id=vN9fpfqoP1. 2267
- 142. Gan, J., Zhong, P., Du, Y., Zhu, Y., Duan, C., Wang, H., Schwalbe-Koda, D., Gomes, 2268 C.P., Persson, K., and Wang, W. (2025). Large language models are innate crystal struc- 2269 ture generators. In AI for Accelerated Materials Design - ICLR 2025. URL: https: 2270 //openreview.net/forum?id=8MFd7wInR5. 2271

- 143. Takahara, I., Mizoguchi, T., and Liu, B. (2025). Accelerated inorganic materials design with 2272 generative ai agents. . URL: https://arxiv.org/abs/2504.00741. arXiv:2504.00741. 2273
- 144. Pan, E., Karpovich, C., and Olivetti, E. (2022). Deep reinforcement learning 2274 for inverse inorganic materials design. . URL: https://arxiv.org/abs/2210.11931. 2275 arXiv:2210.11931. 2276
- 145. Cleeton, C., and Sarkisov, L. (2025). Inverse design of metal-organic frameworks using 2277 deep dreaming approaches. Nature Communications 16, 4806. 2278
- 146. Fung, V., Zhang, J., Hu, G., Ganesh, P., and Sumpter, B.G. (2021). Inverse design of two- 2279 dimensional materials with invertible neural networks. . URL: https://arxiv.org/abs/ 2280 2106.03013. arXiv:2106.03013. 2281
- 147. McDermott, M.J., Dwaraknath, S.S., and Persson, K.A. (2021). A graph-based network for 2282 predicting chemical reaction pathways in solid-state materials synthesis. Nature commu- 2283 nications 12, 3097. 2284
- 148. Huo, H., Bartel, C.J., He, T., Trewartha, A., Dunn, A., Ouyang, B., Jain, A., and Ceder, G. 2285

(2022). Machine-learning rationalization and prediction of solid-state synthesis conditions. 2286 Chemistry of Materials 34, 7323–7336. 2287

- 149. He, T., Huo, H., Bartel, C.J., Wang, Z., Cruse, K., and Ceder, G. (2023). Precursor recom- 2288 mendation for inorganic synthesis by machine learning materials similarity from scientific 2289 literature. Science advances 9, eadg8180. 2290
- 150. MacLeod, B.P., Parlane, F.G., Rupnow, C.C., Dettelbach, K.E., Elliott, M.S., Morrissey, T.D., 2291 Haley, T.H., Proskurin, O., Rooney, M.B., Taherimakhsousi, N. et al. (2022). A self-driving 2292 laboratory advances the pareto front for material properties. Nature communications 13, 2293

995. 2294

- 151. Chitturi, S.R., Ramdas, A., Wu, Y., Rohr, B., Ermon, S., Dionne, J., Jornada, F.H.d., Dunne, 2295 M., Tassone, C., Neiswanger, W. et al. (2024). Targeted materials discovery using bayesian 2296 algorithm execution. npj Computational Materials 10, 156. 2297
- 152. Zhang, J., Liu, B., Liu, Z., Wu, J., Arnold, S., Shi, H., Osterrieder, T., Hauch, J.A., Wu, 2298 Z., Luo, J. et al. (2023). Optimizing perovskite thin-film parameter spaces with machine 2299 learning-guided robotic platform for high-performance perovskite solar cells. Advanced 2300 Energy Materials 13, 2302594. 2301
- 153. Fare, C., Fenner, P., Benatan, M., Varsi, A., and Pyzer-Knapp, E.O. (2022). A multi-fidelity 2302 machine learning approach to high throughput materials screening. npj Computational 2303 Materials 8, 257. 2304
- 154. Merchant, A., Batzner, S., Schoenholz, S.S., Aykol, M., Cheon, G., and Cubuk, E.D. 2305

(2023). Scaling deep learning for materials discovery. Nature 624, 80–85. doi: 10.1038/ 2306 s41586-023-06735-9. 2307

- 155. Lan, T., Wang, H., and An, Q. (2024). Enabling high throughput deep reinforcement learn- 2308 ing with first principles to investigate catalytic reaction mechanisms. Nature Communica- 2309 tions 15, 6281. 2310
- 156. Kulichenko, M., Nebgen, B., Lubbers, N., Smith, J.S., Barros, K., Allen, A.E., Habib, A., 2311 Shinkle, E., Fedik, N., Li, Y.W. et al. (2024). Data generation for machine learning inter- 2312 atomic potentials and beyond. Chemical Reviews 124, 13681–13714. 2313

- 157. Abolhasani, M., and Kumacheva, E. (2023). The rise of self-driving labs in chemical and 2314 materials sciences. Nature Synthesis 2, 483–492. 2315
- 158. Seifrid, M., Pollice, R., Aguilar-Granda, A., Morgan Chan, Z., Hotta, K., Ser, C.T., Vestfrid, 2316 J., Wu, T.C., and Aspuru-Guzik, A. (2022). Autonomous chemical experiments: Challenges 2317 and perspectives on establishing a self-driving lab. Accounts of Chemical Research 55, 2318 2454–2466. 2319
- 159. Chanussot, L., Das, A., Goyal, S., Lavril, T., Shuaibi, M., Riviere, M., Tran, K., Heras- 2320 Domingo, J., Ho, C., Hu, W. et al. (2021). The open catalyst 2020 (OC20) dataset and 2321 community challenges. ACS Catalysis 11, 6059–6072. doi: 10.1021/acscatal.0c04525. 2322
- 160. Szymanski, N.J., Fu, S., Persson, E., and Ceder, G. (2024). Integrated analysis of x-ray 2323 diffraction patterns and pair distribution functions for machine-learned phase identification. 2324 Npj computational materials 10, 45. 2325
- 161. Gibson, J., Hire, A., and Hennig, R.G. (2022). Data-augmentation for graph neural network 2326 learning of the relaxed energies of unrelaxed structures. npj Computational Materials 8, 2327

211. 2328

- 162. Pilania, G., Gubernatis, J., and Lookman, T. (2017). Multi-fidelity machine learning models 2329 for accurate bandgap predictions of solids. Computational Materials Science 129, 156– 2330

163. doi: 10.1016/j.commatsci.2016.12.004. 2331

- 163. De Breuck, P.P., Evans, M.L., and Rignanese, G.M. (2022). Accurate experimental band 2332 gap predictions with multifidelity correction learning. Journal of Materials Informatics 2, 2333

10. doi: 10.20517/jmi.2022.13. 2334

- 164. Guo, Z., Zhang, C., Yu, W., Herr, J., Wiest, O., Jiang, M., and Chawla, N.V. (2021). Few- 2335 shot graph learning for molecular property prediction. In Proceedings of the web confer- 2336 ence 2021. pp. 2559–2567. 2337
- 165. Qian, X., Ju, B., Shen, P., Yang, K., Li, L., and Liu, Q. (2024). Meta learning with attention 2338 based fp-gnns for few-shot molecular property prediction. ACS omega 9, 23940–23948. 2339
- 166. Zhang, R., Wu, C., Yang, Q., Liu, C., Wang, Y., Li, K., Huang, L., and Zhou, F. (2024). 2340 Molfescue: enhancing molecular property prediction in data-limited and imbalanced con- 2341 texts using few-shot and contrastive learning. Bioinformatics 40, btae118. 2342
- 167. Chen, Y., Lu, L., Karniadakis, G.E., and Dal Negro, L. (2020). Physics-informed neural net- 2343 works for inverse problems in nano-optics and metamaterials. Optics express 28, 11618– 2344

11633. 2345

- 168. Fang, Z., and Zhan, J. (2019). Deep physical informed neural networks for metamaterial 2346 design. Ieee Access 8, 24506–24513. 2347
- 169. Zhang, E., Dao, M., Karniadakis, G.E., and Suresh, S. (2022). Analyses of internal struc- 2348 tures and defects in materials using physics-informed neural networks. Science advances 2349 8, eabk0644. 2350
- 170. Wang, A.Y.T., Kauwe, S.K., Murdock, R.J., and Sparks, T.D. (2021). Compositionally re- 2351 stricted attention-based network for materials property predictions. npj Computational Ma- 2352 terials 7, 77. doi: 10.1038/s41524-021-00545-1. 2353

- 171. Chen, C., Ye, W., Zuo, Y., Zheng, C., and Ong, S.P. (2019). Graph networks as a universal 2354 machine learning framework for molecules and crystals. Chemistry of materials 31, 3564– 2355

3572. doi: 10.1021/acs.chemmater.9b01294. 2356

- 172. Zhao, X., Greenberg, J., McClellan, S., Hu, Y.J., Lopez, S., Saikin, S.K., Hu, X., and An, 2357 Y. (2021). Knowledge graph-empowered materials discovery. In 2021 IEEE International 2358 Conference on Big Data (Big Data). IEEE pp. 4628–4632. 2359
- 173. Fang, Y., Zhang, Q., Yang, H., Zhuang, X., Deng, S., Zhang, W., Qin, M., Chen, Z., Fan, 2360 X., and Chen, H. (2022). Molecular contrastive learning with chemical element knowledge 2361 graph. In Proceedings of the AAAI conference on artificial intelligence vol. 36. pp. 3968– 2362

3976. 2363

- 174. Fang, Y., Zhang, Q., Zhang, N., Chen, Z., Zhuang, X., Shao, X., Fan, X., and Chen, H. 2364

(2023). Knowledge graph-enhanced molecular contrastive learning with functional prompt. 2365 Nature Machine Intelligence 5, 542–553. 2366

- 175. Huang, S., and Cole, J.M. (2022). Batterybert: A pretrained language model for battery 2367 database enhancement. Journal of chemical information and modeling 62, 6365–6377. 2368
- 176. Zhao, J., Huang, S., and Cole, J.M. (2023). Opticalbert and opticaltable-sqa: text-and 2369 table-based language models for the optical-materials domain. Journal of chemical infor- 2370 mation and modeling 63, 1961–1981. 2371
- 177. Ock, J., Montoya, J., Schweigert, D., Hung, L., Suram, S.K., and Ye, W. (2024). 2372 Unimat: Unifying materials embeddings through multi-modal learning. arXiv preprint 2373 arXiv:2411.08664. 2374
- 178. Gupta, T., Zaki, M., Krishnan, N.A., and Mausam (2022). MatSciBERT: a materials domain 2375 language model for text mining and information extraction. npj Computational Materials 8, 2376

102. doi: 10.1038/s41524-022-00784-w. 2377

- 179. Chen, C., and Ong, S.P. (2022). A universal graph deep learning interatomic potential for 2378 the periodic table. Nature Computational Science 2, 718–728. URL: https://www.nature. 2379 com/articles/s43588-022-00349-3. doi: 10.1038/s43588-022-00349-3. 2380
- 180. Xing, R., Yao, H., Xi, Z., Sun, M., Li, Q., Tian, J., Wang, H., Xu, D., Ma, Z., and 2381 Zhao, L. (2025). Interpretable x-ray diffraction spectra analysis using confidence evalu- 2382 ated deep learning enhanced by template element replacement. npj Computational Mate- 2383 rials 11. URL: https://www.nature.com/articles/s41524-025-01743-x. doi: 10.1038/ 2384 s41524-025-01743-x. 2385
- 181. Moro, V., Loh, C., Dangovski, R., Ghorashi, A., Ma, A., Chen, Z., Kim, S., Lu, P.Y., Chris- 2386 tensen, T., and Soljaciˇ c,´ M. (2025). Multimodal foundation models for material property 2387 prediction and discovery. Newton 1, 100016. URL: https://doi.org/10.1016/j.newton. 2388 2025.100016. doi: 10.1016/j.newton.2025.100016. 2389
- 182. Pai, C.W., HSUEH, H.W., and han Hsu, S. (2025). Reliability of deep learning models 2390 for scanning electron microscopy analysis. In AI for Accelerated Materials Design - ICLR 2391

2025. URL: https://openreview.net/forum?id=25SZ4g8rpT. 2392

- 183. Ouyang, R., Curtarolo, S., Ahmetcik, E., Scheffler, M., and Ghiringhelli, L.M. (2018). Sisso: 2393 A compressed-sensing method for identifying the best low-dimensional descriptor in an im- 2394 mensity of offered candidates. Phys. Rev. Mater. 2, 083802. URL: https://link.aps.org/ 2395 doi/10.1103/PhysRevMaterials.2.083802. doi: 10.1103/PhysRevMaterials.2.083802. 2396

- 184. Xu, Y., and Qian, Q. (2022). i-sisso: Mutual information-based improved sure independent 2397 screening and sparsifying operator algorithm. Engineering Applications of Artificial In- 2398 telligence 116, 105442. URL: https://www.sciencedirect.com/science/article/pii/ 2399 S0952197622004328. doi: https://doi.org/10.1016/j.engappai.2022.105442. 2400
- 185. Muthyala, M., Sorourifar, F., and Paulson, J.A. (2024). Torchsisso: A pytorch-based im- 2401 plementation of the sure independence screening and sparsifying operator for efficient 2402 and interpretable model discovery. Digital Chemical Engineering 13, 100198. URL: 2403 http://dx.doi.org/10.1016/j.dche.2024.100198. doi: 10.1016/j.dche.2024.100198. 2404
- 186. Jain, S., and Wallace, B.C. (2019). Attention is not Explanation. In J. Burstein, C. Do- 2405 ran, and T. Solorio, eds. Proceedings of the 2019 Conference of the North American 2406 Chapter of the Association for Computational Linguistics: Human Language Technolo- 2407 gies, Volume 1 (Long and Short Papers). Minneapolis, Minnesota: Association for Com- 2408 putational Linguistics pp. 3543–3556. URL: https://aclanthology.org/N19-1357/. doi: 2409 10.18653/v1/N19-1357. 2410
- 187. Kotobi, A., Singh, K., Hoche,¨ D., Bari, S., Meißner, R.H., and Bande, A. (2023). Inte- 2411 grating explainability into graph neural network models for the prediction of x-ray ab- 2412 sorption spectra. Journal of the American Chemical Society 145, 22584–22598. URL: 2413 https://doi.org/10.1021/jacs.3c07513. doi: 10.1021/jacs.3c07513. 2414
- 188. Das, K., Samanta, B., Goyal, P., Lee, S.C., Bhattacharjee, S., and Ganguly, N. (2022). 2415 Crysxpp: An explainable property predictor for crystalline materials. npj Computational 2416 Materials 8, 43. URL: https://doi.org/10.1038/s41524-022-00716-8. doi: 10.1038/ 2417 s41524-022-00716-8. 2418
- 189. Batzner, S., Musaelian, A., Sun, L., Geiger, M., Mailoa, J.P., Kornbluth, M., Molinari, N., 2419 Smidt, T.E., and Kozinsky, B. (2022). E(3)-equivariant graph neural networks for data- 2420 efficient and accurate interatomic potentials. Nature Communications 13, 2453. URL: 2421 https://doi.org/10.1038/s41467-022-29939-5. doi: 10.1038/s41467-022-29939-5. 2422
- 190. Martonova,´ D., Goriely, A., and Kuhl, E. (2025). Generalized invariants meet constitutive 2423 neural networks: A novel framework for hyperelastic materials. Journal of the Mechanics 2424 and Physics of Solids pp. 106352. 2425
- 191. Ramprasad, R., Batra, R., Pilania, G., Mannodi-Kanakkithodi, A., and Kim, C. (2017). 2426 Machine learning in materials informatics: recent applications and prospects. npj Compu- 2427 tational Materials 3, 54. 2428
- 192. Zhao, Z., Ma, D., Chen, L., Sun, L., Li, Z., Xia, Y., Chen, B., Xu, H., Zhu, Z., Zhu, S. et al. 2429

(2025). Developing chemdfm as a large language foundation model for chemistry. Cell 2430 Reports Physical Science 6. 2431

- 193. Kazeev, N., Nong, W., Romanov, I., Zhu, R., Ustyuzhanin, A., Yamazaki, S., and Hippal- 2432 gaonkar, K. (2025). Wyckoff transformer: Generation of symmetric crystals. arXiv preprint 2433 arXiv:2503.02407. 2434
- 194. Antunes, L.M., Butler, K.T., and Grau-Crespo, R. (2023). Crystal structure generation with 2435 autoregressive large language modeling. arXiv preprint arXiv:2307.04340. 2436
- 195. Song, Z., Lu, S., Ju, M., Zhou, Q., and Wang, J. (2025). Accurate prediction of synthesiz- 2437 ability and precursors of 3d crystal structures via large language models. Nature Commu- 2438 nications 16, 6530. 2439

- 196. Fung, V., Zhang, J., Juarez, E., and Sumpter, B.G. (2021). Benchmarking graph neural 2440 networks for materials chemistry. npj Computational Materials 7, 84. 2441
- 197. Takeda, S., Priyadarsini, I., Kishimoto, A., Shinohara, H., Hamada, L., Masataka, H., 2442 Fuchiwaki, J., and Nakano, D. (2023). Multi-modal foundation model for material design. 2443 In AI for Accelerated Materials Design-NeurIPS 2023 Workshop. 2444
- 198. Schmidt, J., Shi, J., Borlido, P., Chen, L., Botti, S., and Marques, M.A. (2017). Predicting 2445 the thermodynamic stability of solids combining density functional theory and machine 2446 learning. Chemistry of Materials 29, 5090–5103. 2447
- 199. Hooshmand, M.J., Sakib-Uz-Zaman, C., and Khondoker, M.A.H. (2023). Machine learn- 2448 ing algorithms for predicting mechanical stiffness of lattice structure-based polymer foam. 2449 Materials 16, 7173. 2450
- 200. Zhang, T., Cai, G., Liu, S., and Puppala, A.J. (2017). Investigation on thermal characteris- 2451 tics and prediction models of soils. International Journal of Heat and Mass Transfer 106, 2452 1074–1086. 2453
- 201. Riebesell, J., Goodall, R.E., Benner, P., Chiang, Y., Deng, B., Ceder, G., Asta, M., Lee, 2454 A.A., Jain, A., and Persson, K.A. (2025). A framework to evaluate machine learning crystal 2455 stability predictions. Nature Machine Intelligence 7, 836–847. 2456
- 202. Goodall, R.E., and Lee, A.A. (2020). Predicting materials properties without crystal struc- 2457 ture: deep representation learning from stoichiometry. Nature communications 11, 6280. 2458 doi: 10.1038/s41467-020-19964-7. 2459
- 203. Tshitoyan, V., Dagdelen, J., Weston, L., Dunn, A., Rong, Z., Kononova, O., Persson, K.A., 2460 Ceder, G., and Jain, A. (2019). Unsupervised word embeddings capture latent knowledge 2461 from materials science literature. Nature 571, 95–98. doi: 10.1038/s41586-019-1335-8. 2462
- 204. Alampara, N., Miret, S., and Jablonka, K.M. (2024). Mattext: Do language models need 2463 more than text & scale for materials modeling? arXiv preprint arXiv:2406.17295. 2464
- 205. Sch¨utt, K.T., Glawe, H., Brockherde, F., Sanna, A., M¨uller, K.R., and Gross, E.K. (2014). 2465 How to represent crystal structures for machine learning: Towards fast prediction of elec- 2466 tronic properties. Physical Review B 89, 205118. 2467
- 206. Srinivas, S.S., and Runkana, V. (2024). Cross-modal learning for chemistry prop- 2468 erty prediction: Large language models meet graph machine learning. arXiv preprint 2469 arXiv:2408.14964. 2470
- 207. Dunn, A., Wang, Q., Ganose, A., Dopp, D., and Jain, A. (2020). Benchmarking materials 2471 property prediction methods: the Matbench test set and Automatminer reference algo- 2472 rithm. npj Computational Materials 6, 138. doi: 10.1038/s41524-020-00406-3. 2473
- 208. Rubungo, A.N., Arnold, C., Rand, B.P., and Dieng, A.B. (2023). Llm-prop: Predicting physi- 2474 cal and electronic properties of crystalline solids from their text descriptions. arXiv preprint 2475 arXiv:2310.14029. 2476
- 209. Tang, Y., Xu, W., Cao, J., Gao, W., Farrell, S., Erichson, B., Mahoney, M.W., Nonaka, 2477 A., and Yao, Z. (2025). Matterchat: A multi-modal llm for material science. arXiv preprint 2478 arXiv:2502.13107. 2479

- 210. Jo, J., Choi, E., Kim, M., and Min, K. (2021). Machine learning-aided materials design 2480 platform for predicting the mechanical properties of na-ion solid-state electrolytes. ACS 2481 Applied Energy Materials 4, 7862–7869. 2482
- 211. Kandavalli, M., Agarwal, A., Poonia, A., Kishor, M., and Ayyagari, K.P.R. (2023). Design of 2483 high bulk moduli high entropy alloys using machine learning. Scientific Reports 13, 20504. 2484
- 212. (2024). Llamp: Large language model made powerful for high-fidelity materials knowledge 2485 retrieval and distillation. arXiv preprint arXiv:2401.17244. 2486
- 213. Chaib, H., Mohammedi, L., Benmebrouk, L., Boukraa, A., Daoudi, B., and Achouri, A. 2487

(2020). Effect of metal atom substitutions in li based hydrides for hydrogen storage. Inter- 2488 national Journal of Hydrogen Energy 45, 28920–28929. 2489

- 214. Goodall, R.E., and Lee, A.A. (2024). Pretraining strategies for structure agnostic material 2490 property prediction. Journal of Chemical Information and Modeling. 2491
- 215. Dan, Y., Jha, D., Gupta, A., and Allu, A. (2024). Pretraining strategies for structure agnostic 2492 material property prediction. Journal of Chemical Information and Modeling 64, 1625– 2493

1635. 2494

- 216. Wang, X., Sheng, Y., Ning, J., Xi, J., Xi, L., Qiu, D., Yang, J., and Ke, X. (2023). A crit- 2495 ical review of machine learning techniques on thermoelectric materials. The Journal of 2496 Physical Chemistry Letters 14, 1808–1822. 2497
- 217. Ziletti, A., Kumar, D., Scheffler, M., and Ghiringhelli, L.M. (2018). Insightful classification of 2498 crystal structures using deep learning. Nature communications 9, 2775. 2499
- 218. Schilling-Wilhelmi, M., R´ıos-Garc´ıa, M., Shabih, S., Gil, M.V., Miret, S., Koch, C.T., 2500 Marquez,´ J.A., and Jablonka, K.M. (2025). From text to insight: large language models 2501 for chemical data extraction. Chemical Society Reviews. 2502
- 219. Paruchuri, A., Wang, Y., Gu, X., and Jayaraman, A. (2024). Machine learning for analyz- 2503 ing atomic force microscopy (afm) images generated from polymer blends. Digital Dis- 2504 covery 3, 2533–2550. URL: https://pubs.rsc.org/en/content/articlehtml/2024/dd/ 2505 d4dd00215f. doi: 10.1039/D4DD00215F. First published 21 Oct 2024. 2506
- 220. (2023). A review of in situ defect detection and monitoring technologies in selective laser 2507 melting. 3D Printing and Additive Manufacturing. URL: https://www.liebertpub.com/ 2508 doi/10.1089/3dp.2021.0114. doi: 10.1089/3dp.2021.0114. Published online 2023-06- 2509 07; add volume/issue/pages and full author list if needed from the publisher citation ex- 2510 port/PDF. 2511
- 221. Fang, Q., Xiong, G., Zhou, M., Tamir, T.S., Yan, C.B., Wu, H., Shen, Z., and Wang, F.Y. 2512

(2022). Process monitoring, diagnosis and control of additive manufacturing. IEEE Trans- 2513 actions on Automation Science and Engineering 21, 1041–1067. 2514

- 222. Pak, P., and Barati Farimani, A. (2025). Additivellm: Large language models predict defects 2515 in metals additive manufacturing. Available at SSRN 5144227. 2516
- 223. Biswas, A., Rade, J., Masud, N., Hasib, M.H.H., Balu, A., Zhang, J., Sarkar, S., Krishna- 2517 murthy, A., Ren, J., and Sarkar, A. (2025). Conversational llm-based decision support for 2518 defect classification in afm images. IEEE Open Journal of Instrumentation and Measure- 2519 ment. 2520

- 224. Ramos, G., Meek, C., Simard, P., Suh, J., and Ghorashi, S. (2020). Interactive machine 2521 teaching: a human-centered approach to building machine-learned models. Human– 2522 Computer Interaction 35, 413–451. 2523
- 225. Shahhosseini, F., Marioriyad, A., Momen, A., Baghshah, M.S., Rohban, M.H., and Ja- 2524 vanmard, S.H. (2025). Large language models for scientific idea generation: A creativity- 2525 centered survey. arXiv:2511.07448. 2526
- 226. Oono, K., and Suzuki, T. (2019). Graph neural networks exponentially lose expressive 2527 power for node classification. arXiv preprint arXiv:1905.10947. 2528
- 227. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, Ł., and 2529 Polosukhin, I. (2017). Attention is all you need. Advances in neural information processing 2530 systems 30. 2531
- 228. Yun, S., Jeong, M., Kim, R., Kang, J., and Kim, H.J. (2019). Graph transformer networks. 2532 Advances in neural information processing systems 32. 2533
- 229. Madani, M., Lacivita, V., Shin, Y., and Tarakanova, A. (2025). Accelerating materials prop- 2534 erty prediction via a hybrid transformer graph framework that leverages four body interac- 2535 tions. npj Computational Materials 11, 15. 2536
- 230. Sanyal, S., Balachandran, J., Yadati, N., Kumar, A., Rajagopalan, P., Sanyal, S., and 2537 Talukdar, P. (2018). Mt-cgcnn: Integrating crystal graph convolutional neural network with 2538 multitask learning for material property prediction. arXiv preprint arXiv:1811.05660. 2539
- 231. Aoki, R., Tung, F., and Oliveira, G.L. (2022). Heterogeneous multi-task learning with expert 2540 diversity. IEEE/ACM Transactions on Computational Biology and Bioinformatics 19, 3093– 2541

3102. 2542

- 232. Shoghi, N., Kolluru, A., Kitchin, J.R., Ulissi, Z.W., Zitnick, C.L., and Wood, B.M. (2023). 2543 From molecules to materials: Pre-training large generalizable models for atomic property 2544 prediction. arXiv preprint arXiv:2310.16802. 2545
- 233. Ruder, S. (2017). An overview of multi-task learning in deep neural networks. arXiv preprint 2546 arXiv:1706.05098. 2547
- 234. Kendall, A., and Gal, Y. (2017). What uncertainties do we need in bayesian deep learning 2548 for computer vision? Advances in neural information processing systems 30. 2549
- 235. Jacobs, R., Goins, P.E., and Morgan, D. (2023). Role of multifidelity data in sequential 2550 active learning materials discovery campaigns: case study of electronic bandgap. Machine 2551 Learning: Science and Technology 4, 045060. 2552
- 236. Xiong, M., Santilli, A., Kirchhof, M., Golinski, A., and Williamson, S. (2024). Efficient and 2553 effective uncertainty quantification for llms. In Neurips Safe Generative AI Workshop 2024. 2554
- 237. Swain, M.C., and Cole, J.M. (2016). Chemdataextractor: a toolkit for automated extraction 2555 of chemical information from the scientific literature. Journal of chemical information and 2556 modeling 56, 1894–1904. 2557
- 238. Weston, L., Tshitoyan, V., Dagdelen, J., Kononova, O., Trewartha, A., Persson, K.A., 2558 Ceder, G., and Jain, A. (2019). Named entity recognition and normalization applied to 2559 large-scale information extraction from the materials science literature. Journal of chemi- 2560 cal information and modeling 59, 3692–3702. 2561

- 239. Kumar, P., Kabra, S., and Cole, J.M. (2025). Mechbert: Language models for extracting 2562 chemical and property relationships about mechanical stress and strain. Journal of Chem- 2563 ical Information and Modeling 65, 1873–1888. 2564
- 240. Polak, M.P., and Morgan, D. (2024). Extracting accurate materials data from research 2565 papers with conversational language models and prompt engineering. Nature Communi- 2566 cations 15, 1569. 2567
- 241. Yi, G.H., Choi, J., Song, H., Miano, O., Choi, J., Bang, K., Lee, B., Sohn, S.S., Buttler, 2568 D., Hiszpanski, A. et al. (2025). Matablegpt: Gpt-based table data extractor from materials 2569 science literature. Advanced Science 12, 2408221. 2570
- 242. da Silva, V.T., Rademaker, A., Lionti, K., Giro, R., Lima, G., Fiorini, S., Archanjo, M., 2571 Carvalho, B.W., Neumann, R., Souza, A. et al. (2024). Automated, llm enabled extrac- 2572 tion of synthesis details for reticular materials from scientific literature. arXiv preprint 2573 arXiv:2411.03484. 2574
- 243. Wang, H., Guo, J., Kong, L., Ramprasad, R., Schwaller, P., Du, Y., and Zhang, C. 2575

(2025). Llm-augmented chemical synthesis and design decision programs. arXiv preprint 2576 arXiv:2505.07027. 2577

- 244. Gupta, S., Mahmood, A., Shetty, P., Adeboye, A., and Ramprasad, R. (2024). Data extrac- 2578 tion from polymer literature using large language models. Communications materials 5, 2579

269. 2580

- 245. Ansari, M., and Moosavi, S.M. (2024). Agent-based learning of materials datasets from 2581 the scientific literature. Digital Discovery 3, 2607–2617. 2582
- 246. Ivanisenko, T.V., Demenkov, P.S., and Ivanisenko, V.A. (2024). An accurate and efficient 2583 approach to knowledge extraction from scientific publications using structured ontology 2584 models, graph neural networks, and large language models. International Journal of 2585 Molecular Sciences 25, 11811. 2586
- 247. Prein, T., Pan, E., Haddouti, S., Lorenz, M., Jehkul, J., Wilk, T., Moran, C., Fotiadis, M.P., 2587 Toshev, A.P., Olivetti, E. et al. (2025). Retro-rank-in: A ranking-based approach for inor- 2588 ganic materials synthesis planning. arXiv preprint arXiv:2502.04289. 2589
- 248. Paulheim, H. (2016). Knowledge graph refinement: A survey of approaches and evaluation 2590 methods. Semantic web 8, 489–508. 2591
- 249. Prince, M.H., Chan, H., Vriza, A., Zhou, T., Sastry, V.K., Luo, Y., Dearing, M.T., Harder, 2592 R.J., Vasudevan, R.K., and Cherukara, M.J. (2024). Opportunities for retrieval and tool 2593 augmented large language models in scientific facilities. npj Computational Materials 10, 2594

251. 2595

- 250. Zhang, R., Zhang, J., Chen, Q., Wang, B., Liu, Y., Qian, Q., Pan, D., Xia, J., Wang, Y., 2596 and Han, Y. (2023). A literature-mining method of integrating text and table extraction for 2597 materials science publications. Computational Materials Science 230, 112441. 2598
- 251. Hira, K., Zaki, M., Sheth, D., Krishnan, N.A. et al. (2024). Reconstructing the materials 2599 tetrahedron: challenges in materials information extraction. Digital Discovery 3, 1021– 2600

1037. 2601

- 252. Zeni, C., Pinsler, R., Z¨ugner, D., Fowler, A., Horton, M., Fu, X., Shysheya, S., Crabbe,´ J., 2602 Sun, L., Smith, J., Nguyen, B., Schulz, H., Lewis, S., Huang, C.W., Lu, Z., Zhou, Y., Yang, 2603 H., Hao, H., Li, J., Tomioka, R., and Xie, T. (2024). Mattergen: a generative model for inor- 2604 ganic materials design. . URL: https://arxiv.org/abs/2312.03687. arXiv:2312.03687. 2605
- 253. (2025). Multi-modal conditional diffusion model using signed distance functions for 2606 metal-organic frameworks generation. Nature Communications 16. URL: https://www. 2607 nature.com/articles/s41467-024-55390-9. doi: 10.1038/s41467-024-55390-9. Pub- 2608 lished 2025-01-01/02; add full author list and article number/pages from the publisher 2609 citation export if required. 2610
- 254. Kim, B., Lee, S., and Kim, J. (2020). Inverse design of porous materials using artificial 2611 neural networks. Science Advances 6, eaax9324. URL: https://www.science.org/doi/ 2612 10.1126/sciadv.aax9324. doi: 10.1126/sciadv.aax9324. 2613
- 255. Metni, H., Ruple, L., Walters, L.N., Torresi, L., Teufel, J., Schopmans, H., Ostreicher,¨ J., 2614 Zhang, Y., Neubert, M., Koide, Y., Steiner, K., Link, P., Bar,¨ L., Petrova, M., Ceder, G., 2615 and Friederich, P. (2025). Generative models for crystalline materials. . URL: https: 2616 //arxiv.org/html/2511.22652v1. arXiv:2511.22652 v1 submitted 2025. 2617
- 256. Xie, T., Fu, X., Ganea, O.E., Barzilay, R., and Jaakkola, T. (2022). Crystal diffusion varia- 2618 tional autoencoder for periodic material generation. . URL: https://arxiv.org/abs/2110. 2619

06197. arXiv:2110.06197. 2620

- 257. Qiu, H., and Sun, Z.Y. (2024). On-demand reverse design of polymers with polytao. npj 2621 Computational Materials 10, 273. 2622
- 258. Zhang, Y., He, X., Gao, S., Zhou, A., and Hao, H. (2024). Evolutionary retrosynthetic route 2623 planning [research frontier]. IEEE Computational Intelligence Magazine 19, 58–72. 2624
- 259. Szymanski, N.J., Rendy, B., Fei, Y., Kumar, R.E., He, T., Milsted, D., McDermott, M.J., 2625 Gallant, M., Cubuk, E.D., Merchant, A. et al. (2023). An autonomous laboratory for the 2626 accelerated synthesis of novel materials. Nature 624, 86–91. 2627
- 260. Yuan, L., Yu, Y., Wei, Y., Wang, Y., Wang, Z., and Wu, F. (2024). Active retrosynthetic 2628 planning aware of route quality. In The Twelfth International Conference on Learning Rep- 2629 resentations. 2630
- 261. Song, X., Pan, X., Zhao, X., Ye, H., Zhang, S., Tang, J., and Yu, T. (2025). Aot*: 2631 Efficient synthesis planning via llm-empowered and-or tree search. arXiv preprint 2632 arXiv:2509.20988. 2633
- 262. Wang, Z., Hutter, F., Zoghi, M., Matheson, D., and de Freitas, N. (2016). Bayesian opti- 2634 mization in a billion dimensions via random embeddings. arXiv preprint arXiv:1301.1942. 2635 URL: https://arxiv.org/abs/1301.1942. arXiv:1301.1942. 2636
- 263. Chang, J., Nikolaev, P., Carpena-N´unez,˜ J., Rao, R., Decker, K., Islam, A.E., Kim, J., Pitt, 2637 M.A., Myung, J.I., and Maruyama, B. (2020). Efficient closed-loop maximization of carbon 2638 nanotube growth rate using bayesian optimization. Scientific reports 10, 9040. 2639
- 264. Li, Y., Cloutier, F., Wu, S., Parviz, A., Knyazev, B., Zhang, Y., Berseth, G., and Liu, 2640 B. (2026). M4olgen:ˆ Multi-agent, multi-stage molecular generation under precise multi- 2641 property constraints. . URL: https://arxiv.org/abs/2601.10131. arXiv:2601.10131. 2642

- 265. Wen, M., Huang, W.F., Dai, J., and Adhikari, S. (2025). Cartesian atomic moment machine 2643 learning interatomic potentials. npj Computational Materials 11, 128. 2644
- 266. Angello, N.H., Friday, D.M., Hwang, C., Yi, S., Cheng, A.H., Torres-Flores, T.C., Jira, E.R., 2645 Wang, W., Aspuru-Guzik, A., Burke, M.D. et al. (2024). Closed-loop transfer enables arti- 2646 ficial intelligence to yield chemical knowledge. Nature 633, 351–358. 2647
- 267. Tom, G., Schmid, S.P., Baird, S.G., Cao, Y., Darvish, K., Hao, H., Lo, S., Pablo-Garc´ıa, 2648 S., Rajaonson, E.M., Skreta, M. et al. (2024). Self-driving laboratories for chemistry and 2649 materials science. Chemical Reviews 124, 9633–9732. 2650
- 268. Burger, B., Maffettone, P.M., Gusev, V.V., Aitchison, C.M., Bai, Y., Wang, X., Li, X., Alston, 2651 B.M., Li, B., Clowes, R., Rankin, N., Harris, B., Sprick, R.S., and Cooper, A.I. (2020). A 2652 mobile robotic chemist. Nature 583, 237–241. 2653
- 269. Epps, R.W., Bowen, M.S., Volk, A.A., Abdel-Latif, K., Han, S., Reyes, K.G., Amassian, A., 2654 and Abolhasani, M. (2020). Artificial chemist: an autonomous quantum dot synthesis bot. 2655 Advanced Materials 32, 2001626. 2656
- 270. Sadeghi, S., Bateni, F., Kim, T., Son, D.Y., Bennett, J.A., Orouji, N., Punati, V.S., Stark, 2657 C., Cerra, T.D., Awad, R. et al. (2024). Autonomous nanomanufacturing of lead-free metal 2658 halide perovskite nanocrystals using a self-driving fluidic lab. Nanoscale 16, 580–591. 2659
- 271. Kusne, A.G., Yu, H., Wu, C., Zhang, H., Hattrick-Simpers, J., DeCost, B., Sarker, S., 2660 Oses, C., Toher, C., Curtarolo, S., Davydov, A.V., Agarwal, R., Bendersky, L.A., Li, M., 2661 Mehta, A., and Takeuchi, I. (2020). On-the-fly closed-loop materials discovery via bayesian 2662 active learning. Nature Communications 11, 5966. URL: https://doi.org/10.1038/ 2663 s41467-020-19597-w. doi: 10.1038/s41467-020-19597-w. 2664
- 272. Szymanski, N.J., Nevatia, P., Bartel, C.J., Zeng, Y., and Ceder, G. (2023). Autonomous and 2665 dynamic precursor selection for solid-state materials synthesis. Nature communications 2666 14, 6956. 2667
- 273. Omidvar, M., Zhang, H., Ihalage, A.A., Saunders, T.G., Giddens, H., Forrester, M., Haq, S., 2668 and Hao, Y. (2024). Accelerated discovery of perovskite solid solutions through automated 2669 materials synthesis and characterization. Nature Communications 15, 6554. 2670
- 274. Boiko, D.A., MacKnight, R., Kline, B., and Gomes, G. (2023). Autonomous chemical re- 2671 search with large language models. Nature 624, 570–578. 2672
- 275. Liu, Y., Checa, M., and Vasudevan, R.K. (2024). Synergizing human expertise and ai effi- 2673 ciency with language model for microscopy operation and automated experiment design. 2674 Machine Learning: Science and Technology 5, 02LT01. 2675
- 276. Xie, Y., He, K., and Castellanos-Gomez, A. (2025). Toward full autonomous laboratory 2676 instrumentation control with large language models. Small Structures pp. 2500173. 2677
- 277. Adams, F., McDannald, A., Takeuchi, I., and Kusne, A.G. (2024). Human-in-the-loop for 2678 bayesian autonomous materials phase mapping. Matter 7, 697–709. 2679
- 278. Biswas, A., Liu, Y., Creange, N., Liu, Y.C., Jesse, S., Yang, J.C., Kalinin, S.V., Ziatdinov, 2680 M.A., and Vasudevan, R.K. (2024). A dynamic bayesian optimized active recommender 2681 system for curiosity-driven partially human-in-the-loop automated experiments. npj Com- 2682 putational Materials 10, 29. 2683

- 279. Xu, P., Ji, X., Li, M., and Lu, W. (2023). Small data machine learning in materials science. 2684 npj Computational Materials 9, 42. 2685
- 280. Medina, J., Ziaullah, A.W., Park, H., Castelli, I.E., Shaon, A., Bensmail, H., and El- 2686 Mellouhi, F. (2022). Accelerating the adoption of research data management strategies. 2687 Matter 5, 3614–3642. 2688
- 281. Ward, L., Dunn, A., Faghaninia, A., Zimmermann, N.E., Bajaj, S., Wang, Q., Montoya, 2689 J.H., Chen, J., Bystrom, K., Dylla, M. et al. (2018). Matminer: an open source toolkit for 2690 materials data mining. Computational Materials Science 152, 60–69. doi: 10.1016/j. 2691 commatsci.2018.05.018. 2692
- 282. Ganose, A.M., Jackson, A.J., and Scanlon, D.O. (2019). Robocrystallographer: automated 2693 crystal structure text descriptions and analysis. MRS Communications 9, 874–881. doi: 2694 10.1557/mrc.2019.94. 2695
- 283. Andersen, C.W., Armiento, R., Blokhin, E., Conduit, G.J., Dwaraknath, S., Evans, M.L., 2696 Akesson,˚ A.,˚ Fekete, A., Gopakumar, A., Graˇzulis, S. et al. (2021). OPTIMADE, an API for 2697 exchanging materials data. Scientific data 8, 217. doi: 10.1038/s41597-021-00974-z. 2698
- 284. Ghiringhelli, L.M., Baldauf, C., Bereau, T., Brockhauser, S., Carbogno, C., Chamanara, J., 2699 Cozzini, S., Curtarolo, S., Draxl, C., Dwaraknath, S. et al. (2023). Shared metadata for 2700 data-centric materials science. Scientific data 10, 626. 2701
- 285. Ramlaoui, A., Siron, M., Djafar, I., Musielewicz, J., Rossello, A., Schmidt, V., and Duval, 2702 A. (2025). LeMat-Traj: A scalable and unified dataset of materials trajectories for atomistic 2703 modeling. arXiv preprint arXiv:2508.20875. arXiv:2508.20875. 2704
- 286. Dinic, F., Wang, Z., Neporozhnii, I., Salim, U.B., Bajpai, R., Rajiv, N., Chavda, V., Radhakr- 2705 ishnan, V., and Voznyy, O. (2023). Strain data augmentation enables machine learning of 2706 inorganic crystal geometry optimization. Patterns 4. 2707
- 287. Chung, J., Zhang, J., Saimon, A.I., Liu, Y., Johnson, B.N., and Kong, Z. (2024). Imbalanced 2708 spectral data analysis using data augmentation based on the generative adversarial net- 2709 work. Scientific Reports 14, 13230. 2710
- 288. Li, W., Chen, Y., Qiu, J., and Wang, X. (2025). MatWheel: Addressing data 2711 scarcity in materials science through synthetic data. arXiv preprint arXiv:2504.09152. 2712 arXiv:2504.09152. 2713
- 289. Li, Q., Fu, N., Omee, S.S., and Hu, J. (2024). Md-hit: Machine learning for material prop- 2714 erty prediction with dataset redundancy control. npj Computational Materials 10, 245. 2715
- 290. Choudhary, K., and DeCost, B. (2021). Atomistic line graph neural network for improved 2716 materials property predictions. npj Computational Materials 7. URL: https://doi.org/10. 2717 1038/s41524-021-00650-1. doi: 10.1038/s41524-021-00650-1. Published 15 November 2718

2021. 2719

- 291. Batzner, S., Musaelian, A., Sun, L., Geiger, M., Mailoa, J.P., Kornbluth, M., Molinari, N., 2720 Smidt, T.E., and Kozinsky, B. (2022). E(3)-equivariant graph neural networks for data- 2721 efficient and accurate interatomic potentials. Nature communications 13, 2453. doi: 10. 2722 1038/s41467-022-29939-5. 2723

- 292. Musaelian, A., Batzner, S., Johansson, A., Sun, L., Owen, C.J., Kornbluth, M., and Kozin- 2724 sky, B. (2023). Learning local equivariant representations for large-scale atomistic dynam- 2725 ics. Nature communications 14, 579. doi: 10.1038/s41467-023-36329-y. 2726
- 293. Huang, C., Chen, C., Shi, L., and Chen, C. (2024). Material property prediction with ele- 2727 ment attribute knowledge graphs and multimodal representation learning. arXiv preprint 2728 arXiv:2411.08414. 2729
- 294. An, Y., Greenberg, J., Kalinowski, A., Zhao, X., Hu, X., Uribe-Romo, F.J., Langlois, K., 2730 Furst, J., and Gomez-Gualdr´ on,´ D.A. (2023). Knowledge graph question answering for 2731 materials science (kgqa4mat): developing natural language interface for metal-organic 2732 frameworks knowledge graph (mof-kg) using llm. arXiv preprint arXiv:2309.11361. 2733
- 295. Shetty, P., Rajan, A.C., Kuenneth, C., Gupta, S., Panchumarti, L.P., Holm, L., Zhang, C., 2734 and Ramprasad, R. (2023). A general-purpose material property data extraction pipeline 2735 from large polymer corpora using natural language processing. npj Computational Mate- 2736 rials 9, 52. doi: 10.1038/s41524-023-01003-w. 2737
- 296. Kuenneth, C., and Ramprasad, R. (2023). polybert: a chemical language model to enable 2738 fully machine-driven ultrafast polymer informatics. Nature communications 14, 4099. 2739
- 297. Tian, S., Jiang, X., Wang, W., Jing, Z., Zhang, C., Zhang, C., Lookman, T., and Su, Y. 2740

(2025). Steel design based on a large language model. Acta Materialia 285, 120663. 2741

- 298. Cavanagh, J.M., Sun, K., Gritsevskiy, A., Bagni, D., Wang, Y., Bannister, T.D., and Head- 2742 Gordon, T. (2025). Smileyllama: Modifying large language models for directed chemical 2743 space exploration. . URL: https://arxiv.org/abs/2409.02231. arXiv:2409.02231. 2744
- 299. Huang, H., Magar, R., and Barati Farimani, A. (2024). Pretraining strategies for structure 2745 agnostic material property prediction. Journal of Chemical Information and Modeling 64, 2746 627–637. 2747
- 300. Kim, J., Kim, Y., Park, J.H., Oh, Y., Kim, S., and Lee, S. (2024). Melt: Materials-aware 2748 continued pre-training for language model adaptation to materials science. arXiv preprint 2749 arXiv:2410.15126. 2750
- 301. First Author, G., Second Author, G., and Others (2025). Matvqa: A visual question an- 2751 swering benchmark for materials science. . URL: https://arxiv.org/abs/2505.18319. 2752 arXiv:2505.18319. 2753
- 302. Wang, G., Xie, Y., Jiang, Y., Mandlekar, A., Xiao, C., Zhu, Y., Fan, L., and Anandkumar, 2754 A. (2023). Voyager: An open-ended embodied agent with large language models. arXiv 2755 preprint arXiv:2305.16291. 2756
- 303. Plaat, A., van Duijn, M., van Stein, N., Preuss, M., van der Putten, P., and Batenburg, K.J. 2757

(2025). Agentic large language models, a survey. arXiv preprint arXiv:2503.23037. 2758

- 304. Song, Y., Miret, S., and Liu, B. (2023). Matsci-nlp: Evaluating scientific language models 2759 on materials science language tasks using text-to-schema modeling. In Proceedings of 2760 the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long 2761 Papers). Toronto, Canada: Association for Computational Linguistics pp. 3570–3595. URL: 2762 https://aclanthology.org/2023.acl-long.201/. 2763

- 305. Ragone, M., Shahbazian-Yassar, R., Mashayek, F., and Yurkiv, V. (2023). Deep learning 2764 modeling in microscopy imaging: A review of materials science applications. Progress in 2765 Materials Science 138, 101165. doi: 10.1016/j.pmatsci.2023.101165. 2766
- 306. Zhao, D., Shi, Z., Liu, Z., Gao, W., Wang, W., Jiang, B., Chen, J., Hu, M., Li, J., Yang, 2767 Y., Yuan, X., Wang, Z., Zhang, Y., Liu, N., Zhao, Y., Yang, B., Zhang, W., Zhao, Y., Li, Y., 2768 Wang, Z., Li, J., Zhang, Q., Lu, S., Yu, H., Wang, Y., Zhang, C., Sun, J., Yang, X., Ma, Z., 2769 Liu, Y., Ye, W., Chai, Z., Li, X., Zhang, L., Zhu, X., Li, G., Song, K., Li, P., Xiong, Y., Xu, 2770 K., Li, H., Li, W., Li, X., and Wang, Z. (2025). Diffractgpt: Atomic structure determination 2771 directly from powder x-ray diffraction patterns. The Journal of Physical Chemistry Let- 2772 ters 16, xxxx–xxxx. URL: https://pubs.acs.org/doi/10.1021/acs.jpclett.4c03137. 2773 doi: 10.1021/acs.jpclett.4c03137. Advance Article; volume/issue/page pending at time 2774 of citation. 2775
- 307. Polat, C., KURBAN, H., Serpedin, E., and Kurban, M. (2025). TDCM25: A multi-modal 2776 multi-task benchmark for temperature-dependent crystalline materials. In AI for Ac- 2777 celerated Materials Design - ICLR 2025. URL: https://openreview.net/forum?id= 2778 bNB5SQTqKL. 2779
- 308. Takeda, S., Priyadarsini, I., Kishimoto, A., Shinohara, H., Hamada, L., Masataka, H., 2780 Fuchiwaki, J., and Nakano, D. (2023). Multi-modal foundation model for material de- 2781 sign. In AI for Accelerated Materials Design - NeurIPS 2023 Workshop. URL: https: 2782 //openreview.net/forum?id=EiT2bLsfM9. 2783
- 309. Odobesku, R., Romanova, K., Mirzaeva, S., Zagorulko, O., Sim, R., Khakimullin, R., Ra- 2784 zlivina, J., Dmitrenko, A., and Vinogradov, V. (2025). nanominer: Multimodal information 2785 extraction for nanomaterials. In AI for Accelerated Materials Design-ICLR 2025. 2786
- 310. Wang, H., Li, W., Jin, X., Cho, K., Ji, H., Han, J., and Burke, M. (2022). Chemical-reaction- 2787 aware molecule representation learning. In Proc. The International Conference on Learn- 2788 ing Representations (ICLR2022). 2789
- 311. Li, X., Wang, L., Luo, Y., Edwards, C., Gui, S., Lin, Y., Ji, H., and Ji, S. (2024). Geometry 2790 informed tokenization of molecules for language model generation. . URL: https://arxiv. 2791 org/abs/2408.10120. arXiv:2408.10120. 2792
- 312. Edwards, C., Han, C., Lee, G., Nguyen, T., Szymkuc, S., Prasad, C.K., Jin, B., Han, J., 2793 Diao, Y., Liu, G., Peng, H., Grzybowski, B.A., Burke, M.D., and Ji, H. (2025). mclm: A 2794 modular chemical language model that generates functional and makeable molecules. In 2795 arxiv. 2796
- 313. Lai, T.M., Zhai, C., and Ji, H. (2023). Knowledge-enhanced biomedical language models. 2797 In Journal of Biomedical Informatics. 2798
- 314. Zhou, M., Fung, Y., Chen, L., Thomas, C., Ji, H., and Chang, S.F. (2023). Enhance chart 2799 understanding via visual language pre-training on plot table pairs. In Proc. The 61st Annual 2800 Meeting of the Association for Computational Linguistics (ACL2023). 2801
- 315. Edwards, C., Lai, T., Ros, K., Honke, G., Cho, K., and Ji, H. (2022). Translation between 2802 molecules and natural language. In Proc. The 2022 Conference on Empirical Methods in 2803 Natural Language Processing (EMNLP2022). 2804

- 316. Nguyen, T., Torres-Flores, T., Hwang, C., Edwards, C., Diao, Y., and Ji, H. (2024). Glad: 2805 Synergizing molecular graphs and language descriptors for enhanced power conversion 2806 efficiency prediction in organic photovoltaic devices. In Proc. 33rd ACM International Con- 2807 ference on Information and Knowledge Management (CIKM 2024). 2808
- 317. Gilmer, J., Schoenholz, S.S., Riley, P.F., Vinyals, O., and Dahl, G.E. (2017). Neural mes- 2809 sage passing for quantum chemistry. In International conference on machine learning. 2810 Pmlr pp. 1263–1272. 2811
- 318. Battaglia, P.W., Hamrick, J.B., Bapst, V., Sanchez-Gonzalez, A., Zambaldi, V., Malinowski, 2812 M., Tacchetti, A., Raposo, D., Santoro, A., Faulkner, R. et al. (2018). Relational inductive 2813 biases, deep learning, and graph networks. arXiv preprint arXiv:1806.01261. 2814
- 319. Scharber, M.C., M¨uhlbacher, D., Koppe, M., Denk, P., Waldauf, C., Heeger, A.J., and 2815 Brabec, C.J. (2006). Design rules for donors in bulk-heterojunction solar cells—towards 2816 10% energy-conversion efficiency. Advanced materials 18, 789–794. 2817
- 320. Brabec, C.J., Dyakonov, V., Parisi, J., and Sariciftci, N.S. (2003). Organic photovoltaics: 2818 concepts and realization vol. 60. Springer Science & Business Media. 2819
- 321. Sch¨utt, K.T., Arbabzadah, F., Chmiela, S., M¨uller, K.R., and Tkatchenko, A. (2017). 2820 Quantum-chemical insights from deep tensor neural networks. Nature communications 2821 8, 13890. 2822
- 322. Mirza, A., Yang, L., Chandran, A.K., Ostreicher,¨ J., Bompas, S., Kazimi, B., Kesselheim, S., 2823 Friederich, P., Sandfeld, S., and Jablonka, K.M. (2025). Matbind: Probing the multimodality 2824 of materials science with contrastive learning. In ICLR AI4Mat Workshop (ICLR 2025). 2825 URL: https://openreview.net/forum?id=ZG0MBXi55v openReview preprint, CC BY 4.0. 2826
- 323. Wu, Y., Ding, M., He, H., Wu, Q., Jiang, S., Zhang, P., and Ji, J. (2025). A ver- 2827 satile multimodal learning framework bridging multiscale knowledge for material de- 2828 sign. npj Computational Materials 11, xxx. URL: https://www.nature.com/articles/ 2829 s41524-025-01767-3. doi: 10.1038/s41524-025-01767-3. Article number pending pagi- 2830 nation at time of citation. 2831
- 324. Alampara, N., Mandal, I., Khetarpal, P., Grover, H.S., Schilling-Wilhelmi, M., Krishnan, 2832 N.M.A., and Jablonka, K.M. (2024). MaCBench: A multimodal chemistry and materials 2833 science benchmark. In AI for Accelerated Materials Design - NeurIPS 2024. URL: https: 2834 //openreview.net/forum?id=Q2PNocDcp6. 2835
- 325. Zhong, X., Gallagher, B., Liu, S., Kailkhura, B., Hiszpanski, A., and Han, T.Y.J. (2022). Ex- 2836 plainable machine learning in materials science. npj Computational Materials 8, 204. URL: 2837 https://doi.org/10.1038/s41524-022-00884-7. doi: 10.1038/s41524-022-00884-7. 2838
- 326. Alvarez-Melis, D., and Jaakkola, T.S. (2018). Towards robust interpretability with self- 2839 explaining neural networks. In Proceedings of the 32nd International Conference on Neu- 2840 ral Information Processing Systems. NIPS’18. Red Hook, NY, USA: Curran Associates Inc. 2841 pp. 7786–7795. 2842
- 327. Agarwal, C., Tanneru, S.H., and Lakkaraju, H. (2024). Faithfulness vs. plausibility: On the 2843 (un)reliability of explanations from large language models. . URL: https://arxiv.org/ 2844 abs/2402.04614. arXiv:2402.04614. 2845

- 328. Lipton, Z. (2018). The mythos of model interpretability: In machine learning, the concept of 2846 interpretability is both important and slippery. Queue 16. doi: 10.1145/3236386.3241340. 2847
- 329. Weng, B., Song, Z., Zhu, R., Yan, Q., Sun, Q., Grice, C.G., Yan, Y., and Yin, W.J. (2020). 2848 Simple descriptor derived from symbolic regression accelerating the discovery of new per- 2849 ovskite catalysts. Nature Communications 11, 3513. URL: https://doi.org/10.1038/ 2850 s41467-020-17263-9. doi: 10.1038/s41467-020-17263-9. 2851
- 330. Naik, R.R., Tiihonen, A., Thapa, J., Batali, C., Liu, Z., Sun, S., and Buonassisi, T. (2022). 2852 Discovering equations that govern experimental materials stability under environmental 2853 stress using scientific machine learning. npj Computational Materials 8, 72. URL: https: 2854 //doi.org/10.1038/s41524-022-00751-5. doi: 10.1038/s41524-022-00751-5. 2855
- 331. Wang, C., Zhang, Y., Wen, C., Yang, M., Lookman, T., Su, Y., and Zhang, 2856 T.Y. (2022). Symbolic regression in materials science via dimension-synchronous- 2857 computation. Journal of Materials Science & Technology 122, 77–83. URL: https:// 2858 www.sciencedirect.com/science/article/pii/S1005030222002055. doi: https://doi. 2859 org/10.1016/j.jmst.2021.12.052. 2860
- 332. Wang, K., Gupta, V., Lee, C.S., Mao, Y., Kilic, M.N.T., Li, Y., Huang, Z., Liao, W.k., Choud- 2861 hary, A., and Agrawal, A. (2024). Xelemnet: towards explainable ai for deep neural net- 2862 works in materials science. Scientific Reports 14, 25178. URL: https://doi.org/10. 2863 1038/s41598-024-76535-2. doi: 10.1038/s41598-024-76535-2. 2864
- 333. Wang, A.Y.T., Kauwe, S.K., Murdock, R.J., and Sparks, T.D. (2021). Compositionally re- 2865 stricted attention-based network for materials property predictions. npj Computational 2866 Materials 7, 77. URL: https://doi.org/10.1038/s41524-021-00545-1. doi: 10.1038/ 2867 s41524-021-00545-1. 2868
- 334. Shi, Y., Zheng, Z., Zhang, Y., Cui, Z., Lin, P., Wen, C., Sa, B., and Sun, Z. (2025). Graph 2869 attention neural networks for interpretable and generalizable prediction of janus iii–vi van 2870 der waals heterostructures. Advanced Intelligent Discovery pp. 202500061. 2871
- 335. Agarwal, C., Queen, O., Lakkaraju, H., and Zitnik, M. (2023). Evaluating explainability 2872 for graph neural networks. Scientific Data 10, 144. URL: https://doi.org/10.1038/ 2873 s41597-023-01974-x. doi: 10.1038/s41597-023-01974-x. 2874
- 336. Sundararajan, M., Taly, A., and Yan, Q. (2017). Axiomatic attribution for deep networks. . 2875 URL: https://arxiv.org/abs/1703.01365. arXiv:1703.01365. 2876
- 337. Dohare, S., Hernandez-Garcia, J.F., Lan, Q., Rahman, P., Mahmood, A.R., and Sutton, 2877 R.S. (2024). Loss of plasticity in deep continual learning. Nature 632, 768–774. 2878
- 338. Springer, J.M., Goyal, S., Wen, K., Kumar, T., Yue, X., Malladi, S., Neubig, G., and Raghu- 2879 nathan, A. (2025). Overtrained language models are harder to fine-tune. arXiv preprint 2880 arXiv:2503.19206. 2881
- 339. Deng, B., Zhong, P., Jun, K., Riebesell, J., Han, K., Bartel, C.J., and Ceder, G. (2023). 2882 Chgnet as a pretrained universal neural network potential for charge-informed atomistic 2883 modelling. Nature Machine Intelligence 5, 1031–1041. URL: https://doi.org/10.1038/ 2884 s42256-023-00716-3. doi: 10.1038/s42256-023-00716-3. 2885

- 340. MacLeod, B.P., Parlane, F.G.L., Morrissey, T.D., Hase,¨ F., Roch, L.M., Dettel- 2886 bach, K.E., Moreira, R., Yunker, L.P.E., Rooney, M.B., Deeth, J.R., Lai, V., Ng, 2887 G.J., Situ, H., Zhang, R.H., Elliott, M.S., Haley, T.H., Dvorak, D.J., Aspuru-Guzik, 2888 A., Hein, J.E., and Berlinguette, C.P. (2020). Self-driving laboratory for acceler- 2889 ated discovery of thin-film materials. Science Advances 6, eaaz8867. URL: https: 2890 //www.science.org/doi/abs/10.1126/sciadv.aaz8867. doi: 10.1126/sciadv.aaz8867. 2891 arXiv:https://www.science.org/doi/pdf/10.1126/sciadv.aaz8867. 2892
- 341. Tom, G., Schmid, S.P., Baird, S.G., Cao, Y., Darvish, K., Hao, H., Lo, S., Pablo-Garc´ıa, 2893 S., Rajaonson, E.M., Skreta, M., Yoshikawa, N., Corapi, S., Akkoc, G.D., Strieth-Kalthoff, 2894 F., Seifrid, M., and Aspuru-Guzik, A. (2024). Self-driving laboratories for chemistry and 2895 materials science. Chemical Reviews 124, 9633–9732. URL: https://doi.org/10.1021/ 2896 acs.chemrev.4c00055. doi: 10.1021/acs.chemrev.4c00055. 2897
- 342. Ghafarollahi, A., and Buehler, M.J. (2024). Sciagents: Automating scientific discovery 2898 through bioinspired multi-agent intelligent graph reasoning. Advanced Materials 37, 2899 e2413523. doi: 10.1002/adma.202413523. 2900
- 343. Choudhary, K., and DeCost, B. (2021). Atomistic line graph neural network for improved 2901 materials property predictions. npj Computational Materials 7, 185. URL: https://doi. 2902 org/10.1038/s41524-021-00650-1. doi: 10.1038/s41524-021-00650-1. 2903
- 344. Jiequn, H., Zhang, L., Roberto, C., and Weinan, E. (2018). Deep potential: A gen- 2904 eral representation of a many-body potential energy surface. Communications in 2905 Computational Physics 23, 629–639. URL: https://global-sci.com/article/80035/ 2906 deep-potential-a-general-representation-of-a-many-body-potential-energy-surface2907. doi: https://doi.org/10.4208/cicp.OA-2017-0213. 2908
- 345. Eriksson, D., Pearce, M., Gardner, J.R., Turner, R., and Poloczek, M. (2019). Scal- 2909 able global optimization via local bayesian optimization. In Advances in Neural Infor- 2910 mation Processing Systems 32 (NeurIPS 2019). URL: https://papers.nips.cc/paper/ 2911 8788-scalable-global-optimization-via-local-bayesian-optimization. 2912
- 346. Qian, X., Yoon, B.J., Arroyave,´ R., Qian, X., and Dougherty, E.R. (2023). Knowledge- 2913 driven learning, optimization, and experimental design under uncertainty for materials dis- 2914 covery. Patterns 4, 100863. URL: https://www.sciencedirect.com/science/article/ 2915 pii/S2666389923002477. doi: https://doi.org/10.1016/j.patter.2023.100863. 2916
- 347. Ramos, M.C., Collison, C.J., and White, A.D. (2025). A review of large language models 2917 and autonomous agents in chemistry. Chemical Science 16, 2514–2572. URL: https:// 2918 www.sciencedirect.com/science/article/pii/S2041652024020650. doi: https://doi. 2919 org/10.1039/d4sc03921a. 2920
- 348. Kahneman, D. (2011). Thinking, Fast and Slow. Farrar, Straus and Giroux. ISBN 2921

0374275637. 2922

- 349. Tom, G., Burger, B., Ghaffari, A. et al. (2024). Self-driving laboratories for chemistry and 2923 materials science. Chemical Reviews 124, 429–480. doi: 10.1021/acs.chemrev.3c00369. 2924
- 350. Volk, A.A., and Abolhasani, M. (2024). Performance metrics to unleash the power of self- 2925 driving labs in chemistry and materials science. Nature Communications 15, 1378. doi: 2926 10.1038/s41467-024-45569-5. 2927

- 351. Gladstone, A., Nanduru, G., Islam, M.M., Han, P., Ha, H., Chadha, A., Du, Y., Ji, H., Li, 2928 J., and Iqbal, T. (2025). Energy-based transformers are scalable learners and thinkers. . 2929 URL: https://arxiv.org/abs/2507.02092. arXiv:2507.02092. 2930
- 352. Dehghannasiri, R., Xue, D., Balachandran, P.V., Yousefi, M.R., Dalton, L.A., Look- 2931 man, T., and Dougherty, E.R. (2017). Optimal experimental design for materials 2932 discovery. Computational Materials Science 129, 311–322. URL: https://www. 2933 sciencedirect.com/science/article/pii/S0927025616306024. doi: https://doi.org/ 2934 10.1016/j.commatsci.2016.11.041. 2935
- 353. Talapatra, A., Boluki, S., Duong, T., Qian, X., Dougherty, E., and Arroyave,´ R. (2018). 2936 Autonomous efficient experiment design for materials discovery with bayesian model 2937 averaging. Phys. Rev. Mater. 2, 113803. URL: https://link.aps.org/doi/10.1103/ 2938 PhysRevMaterials.2.113803. doi: 10.1103/PhysRevMaterials.2.113803. 2939
- 354. Ribeiro, M., Singh, S., and Guestrin, C. (2016). “why should I trust you?”: Explaining 2940 the predictions of any classifier. In J. DeNero, M. Finlayson, and S. Reddy, eds. 2941 Proceedings of the 2016 Conference of the North American Chapter of the Association 2942 for Computational Linguistics: Demonstrations. San Diego, California: Association for 2943 Computational Linguistics pp. 97–101. URL: https://aclanthology.org/N16-3020/. doi: 2944 10.18653/v1/N16-3020. 2945
- 355. Wang, Z., Chen, A., Tao, K., Han, Y., and Li, J. (2024). Matgpt: A vane of materials 2946 informatics from past, present, to future. Advanced Materials 36, 2306733. 2947
- 356. Choi, J., and Lee, B. (2024). Accelerating materials language processing with large 2948 language models. Communications Materials 5, 13. URL: https://doi.org/10.1038/ 2949 s43246-024-00449-9. doi: 10.1038/s43246-024-00449-9. 2950
- 357. Ghafarollahi, A., and Buehler, M.J. (2024). Sciagents: Automating scientific discovery 2951 through multi-agent intelligent graph reasoning. . URL: https://arxiv.org/abs/2409. 2952

05556. arXiv:2409.05556. 2953

- 358. Sim, M., Vakili, M.G., Strieth-Kalthoff, F., Hao, H., Hickman, R.J., Miret, S., Pablo-Garc´ıa, 2954 S., and Aspuru-Guzik, A. (2024). Chemos 2.0: An orchestration architecture for chemical 2955 self-driving laboratories. Matter 7, 2959–2977. 2956
- 359. Roch, L.M., Hase,¨ F., and Aspuru-Guzik, A. (2020). Chemos: An orchestration software 2957 to democratize autonomous discovery. PLoS ONE 15, e0229862. doi: 10.1371/journal. 2958 pone.0229862. 2959
- 360. Panapitiya, G., Saldanha, E., Job, H., and Hess, O. (2025). Autolabs: Cognitive multi- 2960 agent systems with self-correction for autonomous chemical experimentation. . URL: 2961 https://arxiv.org/abs/2509.25651. arXiv:2509.25651. 2962
- 361. Lu, C., Lu, C., Lange, R.T., Foerster, J., Clune, J., and Ha, D. (2024). The ai scientist: 2963 Towards fully automated open-ended scientific discovery. . URL: https://arxiv.org/ 2964 abs/2408.06292. arXiv:2408.06292. 2965
- 362. Beeler, C., Subramanian, S.G., Sprague, K., Chatti, N., Bellinger, C., Shahen, M., Paquin, 2966 N., Baula, M., Dawit, A., Yang, Z., Li, X., Crowley, M., and Tamblyn, I. (2023). Chemgymrl: 2967 An interactive framework for reinforcement learning for digital chemistry. . URL: https: 2968 //arxiv.org/abs/2305.14177. arXiv:2305.14177. 2969

- 363. Montoya, J.H., Winther, K.T., Flores, R.A., Bligaard, T., Hummelshøj, J.S., and Aykol, 2970 M. (2020). Autonomous intelligent agents for accelerated materials discovery. Chemical 2971 Science 11, 8517–8532. 2972
- 364. Allec, S.I., and Ziatdinov, M. (2025). Active and transfer learning with partially bayesian 2973 neural networks for materials and chemicals. Digital Discovery 4, 1284–1297. 2974
- 365. Author, N. (2025). Agentic assistant for material scientists. ChemRxiv. URL: https: 2975 //chemrxiv.org/engage/chemrxiv/article-details/67ef7ed2fa469535b9b23677. doi: 2976 10.26434/chemrxiv.67ef7ed2fa469535b9b23677. Preprint. 2977
- 366. Szymanski, N.J., Rendy, B., Fei, Y., Kumar, R.E., He, T., Milsted, D., McDermott, M.J., Gal- 2978 lant, M., Cubuk, E.D., Merchant, A., Kim, H., Jain, A., Bartel, C.J., Persson, K., Zeng, Y., 2979 and Ceder, G. (2023). An autonomous laboratory for the accelerated synthesis of novel 2980 materials. Nature 624, 86–91. URL: https://doi.org/10.1038/s41586-023-06734-w. 2981 doi: 10.1038/s41586-023-06734-w. 2982
- 367. Nikolaev, P., Hooper, D., Webber, F., Rao, R., Decker, K., Krein, M., Poleski, J., Barto, R., 2983 and Maruyama, B. (2016). Autonomy in materials research: a case study in carbon nan- 2984 otube growth. npj Computational Materials 2, 16031. URL: https://doi.org/10.1038/ 2985 npjcompumats.2016.31. doi: 10.1038/npjcompumats.2016.31. 2986
- 368. Ding, Q., Miret, S., and Liu, B. (2025). Matexpert: Decomposing materials discovery by 2987 mimicking human experts. In The Thirteenth International Conference on Learning Rep- 2988 resentations. URL: https://openreview.net/forum?id=AUBvo4sxVL. 2989
- 369. Snoek, J., Larochelle, H., and Adams, R.P. (2012). Practical bayesian optimization of 2990 machine learning algorithms. In F. Pereira, C. Burges, L. Bottou, and K. Wein- 2991 berger, eds. Advances in Neural Information Processing Systems vol. 25. Curran As- 2992 sociates, Inc. URL: https://proceedings.neurips.cc/paper_files/paper/2012/file/ 2993 05311655a15b75fab86956663e1819cd-Paper.pdf. 2994
- 370. Lakshminarayanan, B., Pritzel, A., and Blundell, C. (2017). Simple and scalable 2995 predictive uncertainty estimation using deep ensembles. In I. Guyon, U.V. 2996 Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Gar- 2997 nett, eds. Advances in Neural Information Processing Systems vol. 30. Curran Asso- 2998 ciates, Inc. URL: https://proceedings.neurips.cc/paper_files/paper/2017/file/ 2999 9ef2ed4b7fd2c810847ffa5fa85bce38-Paper.pdf. 3000
- 371. Bensberg, M., and Reiher, M. (2024). Uncertainty-aware first-principles exploration of 3001 chemical reaction networks. The Journal of Physical Chemistry A 128, 4532–4547. 3002
- 372. Joshi, R.P., and Kumar, N. (2021). Artificial intelligence based autonomous molecular de- 3003 sign for medical therapeutic: A perspective. . URL: https://arxiv.org/abs/2102.06045. 3004 arXiv:2102.06045. 3005
- 373. Kalidindi, S.R., Buzzy, M., Boyce, B.L., and Dingreville, R. (2022). Digital twins for materi- 3006 als. Frontiers in Materials 9, 818535. 3007
- 374. Su, H., Chen, R., Tang, S., Yin, Z., Zheng, X., Li, J., Qi, B., Wu, Q., Li, H., Ouyang, 3008 W., Torr, P., Zhou, B., and Dong, N. (2025). Many heads are better than one: Improved 3009 scientific idea generation by a llm-based multi-agent system. . URL: https://arxiv.org/ 3010 abs/2410.09403. arXiv:2410.09403. 3011

- 375. Vasu, R., Jansen, P., Siangliulue, P., Sarasua, C., Bernstein, A., Clark, P., and Mishra, B.D. 3012

(2025). Harpa: A testability-driven, literature-grounded framework for research ideation. . 3013 URL: https://arxiv.org/abs/2510.00620. arXiv:2510.00620. 3014

- 376. Alkan, A.K., Sourav, S., Jablonska, M., Astarita, S., Chakrabarty, R., Garuda, N., 3015 Khetarpal, P., Pioro,´ M., Tanoglidis, D., Iyer, K.G., Polimera, M.S., Smith, M.J., Ghosal, 3016 T., Huertas-Company, M., Kruk, S., Schawinski, K., and Ciuca,˘ I. (2025). A survey on 3017 hypothesis generation for scientific discovery in the era of large language models. . URL: 3018 https://arxiv.org/abs/2504.05496. arXiv:2504.05496. 3019
- 377. Treloar, N.J., Braniff, N., Ingalls, B., and Barnes, C.P. (2022). Deep reinforcement learning 3020 for optimal experimental design in biology. PLOS Computational Biology 18, e1010695. 3021
- 378. Shneiderman, B. (2020). Human-centered artificial intelligence: Three fresh ideas. AIS 3022 Transactions on Human-Computer Interaction 12, 109–124. URL: https://doi.org/10. 3023 17705/1thci.00131. doi: 10.17705/1thci.00131. 3024
- 379. King, R.D., Rowland, J., Oliver, S.G., Young, M., Aubrey, W., Byrne, E., Liakata, M., 3025 Markham, M., Pir, P., Soldatova, L.N., Sparkes, A., Whelan, K.E., and Clare, A. (2009). The 3026 automation of science. Science 324, 85–89. URL: https://doi.org/10.1126/science. 3027

1165620. doi: 10.1126/science.1165620. 3028

- 380. Tom, G., Schmid, S.P., Baird, S.G., Cao, Y., Darvish, K., Hao, H., Lo, S., Pablo-Garc´ıa, 3029 S., Rajaonson, E.M., Skreta, M., Yoshikawa, N., Corapi, S., Akkoc, G.D., Strieth-Kalthoff, 3030 F., Seifrid, M., and Aspuru-Guzik, A. (2024). Self-driving laboratories for chemistry and 3031 materials science. Chemical Reviews 124, 9633–9732. URL: https://doi.org/10.1021/ 3032 acs.chemrev.4c00055. doi: 10.1021/acs.chemrev.4c00055. 3033
- 381. Bi, J., Xu, Y., Conrad, F., Wiemer, H., and Ihlenfeldt, S. (2025). A comprehen- 3034 sive benchmark of active learning strategies with automl for small-sample regression 3035 in materials science. Scientific Reports 15, 37167. URL: https://doi.org/10.1038/ 3036 s41598-025-24613-4. doi: 10.1038/s41598-025-24613-4. 3037
- 382. Anstine, D.M., and Isayev, O. (2023). Generative models as an emerging paradigm in the 3038 chemical sciences. Journal of the American Chemical Society 145, 8736–8750. URL: 3039 https://doi.org/10.1021/jacs.2c13467. doi: 10.1021/jacs.2c13467. 3040
- 383. Nigam, A., Pollice, R., Krenn, M., Gomes, G.d.P., and Aspuru-Guzik, A. (2021). Beyond 3041 generative models: superfast traversal, optimization, novelty, exploration and discovery 3042 (stoned) algorithm for molecules using selfies. Chemical Science 12, 7079–7090. URL: 3043 https://doi.org/10.1039/D1SC00231G. doi: 10.1039/D1SC00231G. 3044
- 384. Dagdelen, J., Dunn, A., Lee, S., Walker, N., Rosen, A.S., Ceder, G., Persson, K.A., 3045 and Jain, A. (2024). Structured information extraction from scientific text with large lan- 3046 guage models. Nature Communications 15, 1418. URL: https://doi.org/10.1038/ 3047 s41467-024-45563-x. doi: 10.1038/s41467-024-45563-x. 3048
- 385. Boiko, D.A., MacKnight, R., Kline, B., and Gomes, G. (2023). Autonomous chem- 3049 ical research with large language models. Nature 624, 570–578. doi: 10.1038/ 3050 s41586-023-06792-0. 3051
- 386. Amirian, B., Dale, A.S., Kalinin, S., and Hattrick-Simpers, J. (2025). Building trustworthy 3052 ai for materials discovery: From autonomous laboratories to z-scores. . URL: https: 3053 //arxiv.org/abs/2512.01080. arXiv:2512.01080. 3054

- 387. Bommasani, R., Hudson, D.A., Adeli, E., Altman, R., Arora, S., von Arx, S., Bernstein, 3055 M.S., Bohg, J., Bosselut, A., Brunskill, E., Brynjolfsson, E., Buch, S., Card, D., Castellon, 3056 R., Chatterji, N., Chen, A., Creel, K., Davis, J.Q., Demszky, D., Donahue, C., Doumbouya, 3057 M., Durmus, E., Ermon, S., Etchemendy, J., Ethayarajh, K., Fei-Fei, L., Finn, C., Gale, T., 3058 Gillespie, L., Goel, K., Goodman, N., Grossman, S., Guha, N., Hashimoto, T., Henderson, 3059 P., Hewitt, J., Ho, D.E., Hong, J., Hsu, K., Huang, J., Icard, T., Jain, S., Jurafsky, D., Kalluri, 3060 P., Karamcheti, S., Keeling, G., Khani, F., Khattab, O., Koh, P.W., Krass, M., Krishna, R., 3061 Kuditipudi, R., Kumar, A., Ladhak, F., Lee, M., Lee, T., Leskovec, J., Levent, I., Li, X.L., Li, 3062 X., Ma, T., Malik, A., Manning, C.D., Mirchandani, S., Mitchell, E., Munyikwa, Z., Nair, S., 3063 Narayan, A., Narayanan, D., Newman, B., Nie, A., Niebles, J.C., Nilforoshan, H., Nyarko, 3064 J., Ogut, G., Orr, L., Papadimitriou, I., Park, J.S., Piech, C., Portelance, E., Potts, C., 3065 Raghunathan, A., Reich, R., Ren, H., Rong, F., Roohani, Y., Ruiz, C., Ryan, J., Re,´ C., 3066 Sadigh, D., Sagawa, S., Santhanam, K., Shih, A., Srinivasan, K., Tamkin, A., Taori, R., 3067 Thomas, A.W., Tramer,` F., Wang, R.E. et al. (2022). On the opportunities and risks of 3068 foundation models. . URL: https://arxiv.org/abs/2108.07258. arXiv:2108.07258. 3069
- 388. Jiang, S., and Webb, M.A. (2025). Generative active learning across polymer architectures 3070 and solvophobicities for targeted rheological behaviour. npj Computational Materials 11, 3071

90. doi: 10.1038/s41524-025-01900-2. 3072

- 389. Zhang, Y., Khan, S.A., Mahmud, A., Yang, H., Lavin, A., Levin, M., Frey, J., Dunnmon, J., 3073 Evans, J., Bundy, A., Dzeroski, S., Tegner, J., and Zenil, H. (2025). Exploring the role of 3074 large language models in the scientific method: from hypothesis to discovery. npj Artificial 3075 Intelligence 1, 14. URL: https://doi.org/10.1038/s44387-025-00019-5. doi: 10.1038/ 3076 s44387-025-00019-5. 3077
- 390. Regenwetter, L., Obaideh, Y.A., and Ahmed, F. (2024). Mcd: A model-agnostic counter- 3078 factual search method for multi-modal design modifications. . URL: https://arxiv.org/ 3079 abs/2305.11308. arXiv:2305.11308. 3080
- 391. Mandal, I., Soni, J., Zaki, M., Smedskjaer, M.M., Wondraczek, K., Wondraczek, L., 3081 Gosvami, N.N., and Krishnan, N.M.A. (2025). Evaluating large language model agents 3082 for automation of atomic force microscopy. Nature Communications 16, 9104. URL: 3083 https://doi.org/10.1038/s41467-025-64105-7. doi: 10.1038/s41467-025-64105-7. 3084
- 392. Zhong, X., Gallagher, B., Liu, S., Kailkhura, B., Hiszpanski, A., and Han, T.Y.J. (2022). Ex- 3085 plainable machine learning in materials science. npj Computational Materials 8, 204. URL: 3086 https://doi.org/10.1038/s41524-022-00884-7. doi: 10.1038/s41524-022-00884-7. 3087
- 393. Oviedo, F., Ferres, J.L., Buonassisi, T., and Butler, K.T. (2022). Interpretable and ex- 3088 plainable machine learning for materials science and chemistry. Accounts of Materi- 3089 als Research 3, 597–607. URL: https://doi.org/10.1021/accountsmr.1c00244. doi: 3090 10.1021/accountsmr.1c00244. 3091
- 394. Roscher, R., Bohn, B., Duarte, M.F., and Garcke, J. (2020). Explainable machine learn- 3092 ing for scientific insights and discoveries. IEEE Access 8, 42200–42216. doi: 10.1109/ 3093 ACCESS.2020.2976199. 3094
- 395. Lundberg, S.M., and Lee, S.I. (2017). A unified approach to interpreting model predictions. 3095 In I. Guyon, U.V. Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and 3096 R. Garnett, eds. Advances in Neural Information Processing Systems vol. 30. Curran As- 3097 sociates, Inc. URL: https://proceedings.neurips.cc/paper_files/paper/2017/file/ 3098 8a20a8621978632d76c43dfd28b67767-Paper.pdf. 3099

- 396. Wang, Y., Wagner, N., and Rondinelli, J.M. (2019). Symbolic regression in materials sci- 3100 ence. MRS communications 9, 793–805. 3101
- 397. Nair, A.S., Foppa, L., and Scheffler, M. (2025). Materials-discovery workflow guided by 3102 symbolic regression for identifying acid-stable oxides for electrocatalysis. npj Compu- 3103 tational Materials 11, 150. URL: https://doi.org/10.1038/s41524-025-01596-4. doi: 3104 10.1038/s41524-025-01596-4. 3105
- 398. Udrescu, S.M., and Tegmark, M. (2020). Ai feynman: A physics-inspired method for sym- 3106 bolic regression. Science Advances 6, eaay2631. doi: 10.1126/sciadv.aay2631. 3107
- 399. Bommasani, R., Hudson, D.A., Adeli, E., Altman, R., Arora, S., von Arx, S., Bernstein, 3108 M.S., Bohg, J., Bosselut, A., Brunskill, E., Brynjolfsson, E., Buch, S., Card, D., Castellon, 3109 R., Chatterji, N., Chen, A., Creel, K., Davis, J.Q., Demszky, D., Donahue, C., Doumbouya, 3110 M., Durmus, E., Ermon, S., Etchemendy, J., Ethayarajh, K., Fei-Fei, L., Finn, C., Gale, T., 3111 Gillespie, L., Goel, K., Goodman, N., Grossman, S., Guha, N., Hashimoto, T., Henderson, 3112 P., Hewitt, J., Ho, D.E., Hong, J., Hsu, K., Huang, J., Icard, T., Jain, S., Jurafsky, D., Kalluri, 3113 P., Karamcheti, S., Keeling, G., Khani, F., Khattab, O., Koh, P.W., Krass, M., Krishna, R., 3114 Kuditipudi, R., Kumar, A., Ladhak, F., Lee, M., Lee, T., Leskovec, J., Levent, I., Li, X.L., Li, 3115 X., Ma, T., Malik, A., Manning, C.D., Mirchandani, S., Mitchell, E., Munyikwa, Z., Nair, S., 3116 Narayan, A., Narayanan, D., Newman, B., Nie, A., Niebles, J.C., Nilforoshan, H., Nyarko, 3117 J., Ogut, G., Orr, L., Papadimitriou, I., Park, J.S., Piech, C., Portelance, E., Potts, C., 3118 Raghunathan, A., Reich, R., Ren, H., Rong, F., Roohani, Y., Ruiz, C., Ryan, J., Re,´ C., 3119 Sadigh, D., Sagawa, S., Santhanam, K., Shih, A., Srinivasan, K., Tamkin, A., Taori, R., 3120 Thomas, A.W., Tramer,` F., Wang, R.E. et al. (2022). On the opportunities and risks of 3121 foundation models. . URL: https://arxiv.org/abs/2108.07258. arXiv:2108.07258. 3122
- 400. Bender, A., and Cortes-Ciriano,´ I. (2021). Artificial intelligence in drug discovery: what is 3123 realistic, what are illusions? part 1: Ways to make an impact, and why we are not there 3124 yet. Drug discovery today 26, 511–524. 3125
- 401. Tan, Z., Yang, Q., and Luo, S. (2025). Ai molecular catalysis: where are we now? Organic 3126 Chemistry Frontiers 12, 2759–2776. 3127
- 402. Wang, Z., Hou, L., Lu, T., Wu, Y., Li, Y., Yu, H., and Ji, H. (2024). Enabling language 3128 models to implicitly learn self-improvement. . URL: https://arxiv.org/abs/2310.00898. 3129 arXiv:2310.00898. 3130
- 403. Acikgoz, E.C., Qian, C., Ji, H., Hakkani-T¨ur, D., and Tur, G. (2025). Self-improving llm 3131 agents at test-time. In arxiv. 3132
- 404. Gao, H.a., Geng, J., Hua, W., Hu, M., Juan, X., Liu, H., Liu, S., Qiu, J., Qi, X., Wu, Y., 3133 Wang, H., Xiao, H., Zhou, Y., Zhang, S., Zhang, J., Xiang, J., Fang, Y., Zhao, Q., Liu, D., 3134 Ren, Q., Qian, C., Wang, Z., Hu, M., Wang, H., Wu, Q., Ji, H., and Wang, M. (2025). A 3135 survey of self-evolving agents: On path to artificial super intelligence. In arxiv. 3136
- 405. Vaccaro, M., Almaatouq, A., and Malone, T. (2024). When combinations of humans and 3137 ai are useful: A systematic review and meta-analysis. Nature Human Behaviour 8, 2293– 3138

2303. 3139

- 406. Holzinger, A., Carrington, A., and M¨uller, H. (2020). Measuring the quality of explanations: 3140 The system causability scale (scs). KI - K¨unstliche Intelligenz 34, 193–198. URL: https: 3141 //doi.org/10.1007/s13218-020-00636-z. doi: 10.1007/s13218-020-00636-z. 3142

- 407. Singh, S., Hindriks, K., Heylen, D., and Baraka, K. (2025). A systematic review of human-ai 3143 co-creativity. . URL: https://arxiv.org/abs/2506.21333. arXiv:2506.21333. 3144
- 408. Bollaert, M., Augereau, O., and Coppin, G. (2023). Measuring and calibrating trust in 3145 artificial intelligence. In IFIP Conference on Human-Computer Interaction. Springer pp. 3146 232–237. 3147
- 409. Lu, C., Lu, C., Lange, R.T., Foerster, J., Clune, J., and Ha, D. (2024). The ai scientist: 3148 Towards fully automated open-ended scientific discovery. arXiv preprint arXiv:2408.06292. 3149
- 410. Chen, Z., Luo, Y., and Sra, M. (2025). Engaging with ai: How interface design shapes 3150 human-ai collaboration in high-stakes decision-making. . URL: https://arxiv.org/abs/ 3151 2501.16627. arXiv:2501.16627. 3152
- 411. Barabasi,´ A., Jeong, H., Neda,´ Z., Ravasz, E., Schubert, A., and Vicsek, T. (2002). Evo- 3153 lution of the social network of scientific collaborations. Physica A: Statistical Mechanics 3154 and its Applications 311, 590–614. URL: http://dx.doi.org/10.1016/S0378-4371(02) 3155 00736-7. doi: 10.1016/s0378-4371(02)00736-7. 3156
- 412. Langley, P. (1987). Scientific discovery: Computational explorations of the creative pro- 3157 cesses. MIT press. 3158
- 413. Cui, H., and Yasseri, T. (2024). Ai-enhanced collective intelligence. Patterns 5. 3159
- 414. Novelli, C., Taddeo, M., and Floridi, L. (2024). Accountability in artificial intelligence: What 3160 it is and how it works. Ai & Society 39, 1871–1882. 3161
- 415. Oduoye, M.O., Javed, B., Gupta, N., and Sih, C.M.V. (2023). Algorithmic bias and research 3162 integrity; the role of nonhuman authors in shaping scientific knowledge with respect to 3163 artificial intelligence: a perspective. International Journal of Surgery 109, 2987–2990. 3164
- 416. Amirian, B., Dale, A.S., Kalinin, S., and Hattrick-Simpers, J. (2025). Building trustworthy 3165 ai for materials discovery: From autonomous laboratories to z-scores. . URL: https: 3166 //arxiv.org/abs/2512.01080. arXiv:2512.01080. 3167
- 417. Blau, W., Cerf, V.G., Enriquez, J., Francisco, J.S., Gasser, U., Gray, M.L., Greaves, M., 3168 Grosz, B.J., Jamieson, K.H., Haug, G.H. et al. (2024). Protecting scientific integrity in an 3169 age of generative ai. National Academy of Sciences. 3170
- 418. Lo Piano, S. (2020). Ethical principles in machine learning and artificial intelligence: cases 3171 from the field and possible ways forward. Humanities and Social Sciences Communica- 3172 tions 7, 1–7. 3173
- 419. Abel, D., Bowling, M., Barreto, A., Dabney, W., Dong, S., Hansen, S., Harutyunyan, A., 3174 Khetarpal, K., Lyle, C., Pascanu, R. et al. (2025). Plasticity as the mirror of empowerment. 3175 arXiv preprint arXiv:2505.10361. 3176
- 420. Barto, A.G. (2024). In the beginning ml was rl. Video. . URL: https://www.youtube. 3177 com/watch?v=-gQNM7rAWP0 reinforcement Learning Conference (RLC 2024), Oct 1, 2024. 3178 Edited by Gor Baghdasaryan. 3179
- 421. METR (2025). Recent frontier models are reward hacking. https://metr. 3180 org/blog/2025-06-05-recent-reward-hacking/. . URL: https://metr.org/blog/ 3181 2025-06-05-recent-reward-hacking/ blog post. 3182

- 422. Belcak, P., Heinrich, G., Diao, S., Fu, Y., Dong, X., Muralidharan, S., Lin, Y.C., and 3183 Molchanov, P. (2025). Small language models are the future of agentic AI. arXiv preprint 3184 arXiv:2506.02153. arXiv:2506.02153. 3185
- 423. Grosse, R., Bae, J., Anil, C., Elhage, N., Tamkin, A., Tajdini, A., Steiner, B., Li, D., Durmus, 3186 E., Perez, E., Hubinger, E., Lukosiute, K., Nguyen, K., Joseph, N., McCandlish, S., Kaplan, 3187 J., and Bowman, S.R. (2023). Studying large language model generalization with influence 3188 functions. arXiv preprint arXiv:2308.03296. arXiv:2308.03296. 3189
- 424. Oh, J., Farquhar, G., Kemaev, I., Calian, D.A., Hessel, M., Zintgraf, L., Singh, S., van 3190 Hasselt, H., and Silver, D. (2025). Discovering state-of-the-art reinforcement learning al- 3191 gorithms. Nature. 3192
- 425. Malik, S.A., Doherty, T., Tigas, P., Razzak, M., Gal, Y., and Walsh, A. (2025). Towards 3193 dynamic benchmarks for autonomous materials discovery. In AI for Accelerated Materials 3194 Design - NeurIPS 2025. URL: https://openreview.net/forum?id=Cfj7uBu5dy. 3195
- 426. Govindarajan, P., Reymond, M., Clavaud, A., Phielipp, M., Miret, S., and Chandar, S. 3196

(2025). Crystalgym: A new benchmark for materials discovery using reinforcement learn- 3197 ing. arXiv preprint arXiv:2509.23156. 3198

- 427. Lu, S., Wang, Z., Zhang, H., Wu, Q., Gan, L., Zhuang, C., Gu, J., and Lin, T. (2025). Don’t 3199 just fine-tune the agent, tune the environment. arXiv preprint arXiv:2510.10197. 3200
- 428. He, B., Li, H., Jang, Y.K., Jia, M., Cao, X., Shah, A., Shrivastava, A., and Lim, S.N. (2024). 3201 Ma-lmm: Memory-augmented large multimodal model for long-term video understanding. 3202 In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 3203 pp. 13504–13514. 3204
- 429. Yu, H., Chen, T., Feng, J., Chen, J., Dai, W., Yu, Q., Zhang, Y.Q., Ma, W.Y., Liu, J., Wang, 3205 M. et al. (2025). Memagent: Reshaping long-context llm with multi-conv rl-based memory 3206 agent. arXiv preprint arXiv:2507.02259. 3207
- 430. Xu, W., Mei, K., Gao, H., Tan, J., Liang, Z., and Zhang, Y. (2025). A-mem: Agentic memory 3208 for llm agents. arXiv preprint arXiv:2502.12110. 3209
- 431. Guo, T., Chen, X., Wang, Y., Chang, R., Pei, S., Chawla, N.V., Wiest, O., and Zhang, X. 3210

(2024). Large language model based multi-agents: A survey of progress and challenges. 3211 arXiv preprint arXiv:2402.01680. 3212

- 432. Tran, K.T., Dao, D., Nguyen, M.D., Pham, Q.V., O’Sullivan, B., and Nguyen, H.D. (2025). 3213 Multi-agent collaboration mechanisms: A survey of llms. arXiv preprint arXiv:2501.06322. 3214
- 433. Bengio, Y. et al. (2025). Statement on biosecurity risks at the conver- 3215 gence of ai and the life sciences. https://www.nti.org/analysis/articles/ 3216 statement-on-biosecurity-risks-at-the-convergence-of-ai-and-the-life-sciences/3217.

. Nuclear Threat Initiative. Accessed: 2025-08-14. 3218

- 434. Calian, D.A., Farquhar, G., Kemaev, I., Zintgraf, L.M., Hessel, M., Shar, J., Oh, J., Gyorgy,¨ 3219 A., Schaul, T., Dean, J., van Hasselt, H., and Silver, D. (2025). Datarater: Meta-learned 3220 dataset curation. . URL: https://arxiv.org/abs/2505.17895. arXiv:2505.17895. 3221

