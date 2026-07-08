# arXiv:2509.23219v1[cs.LG]27Sep2025

[Figure 1]

## WirelessMathLM: Teaching Mathematical Reasoning for LLMs in Wireless Communications with Reinforcement Learning

#### Xin Li, Mengbing Liu, Yiyang Zhu, Wenhe Zhang Li Wei, Jiancheng An, Chau Yuen

Nanyang Technological University

#### Abstract

Large language models (LLMs) excel at general mathematical reasoning but fail catastrophically on specialized technical mathematics. In wireless communications, where problems require precise manipulation of information-theoretic bounds, optimization constraints, and signal processing formulations, even state-of-the-art models struggle to achieve competent performance. We present WirelessMathLM, demonstrating that compact models (0.5B–7B parameters) can match or exceed much larger models through domain-specific reinforcement learning with verifiable rewards. Our key insight is that wireless mathematics problems possess a unique property—verifiable correctness—that enables effective reinforcement learning without human feedback. We construct WirelessMathBench-XL, a comprehensive benchmark of 4,027 problems from 970 papers. Using Group Relative Policy Optimization (GRPO) with binary verification rewards, we train models directly from base checkpoints without supervised warm-start. Our 7B model achieves 39.5% accuracy on WirelessMathBench-XL, approaching GPT-4o (40.4%) while using ≈100× fewer parameters than DeepSeek-R1 (671B, 57.4%). Remarkably, GRPO training nearly doubles performance across all model scales (0.5B: +11%, 3B: +103%, 7B: +81%), with positive transfer to general mathematics benchmarks—our models gain +8.4 points on average across MATH, Minerva-Math, OlympiadBench, AMC, and AIME without any training on these tasks.

Project Homepage: https://lixin.ai/WirelessMathLM

Date: September 30, 2025

50

57.4 57.9

60

Base Model

39.5 (81% )

53.8 54.2 54.9

+ GRPO (Ours)

40

###### Accuracy(%)

###### Accuracy(%)

42.1

40.4

39.5

40

25.1 (103% )

37.5

30

21.9 14.9 (11% )

25.1

20

20

13.4 12.4

10

0

0

0.5B 3B 7B

Claude-4.0-SonnetGPT-4oGemini-2.5-FlashGrok-4-FastDeepSeek-R1Qwen2.5-72B-InsGPT-5Qwen2.5-Math-72BOurs-3BOurs-7B

Model Size

(a) WirelessMathBench-XL Performance

(b) Impact of GRPO Training

Figure 1 WirelessMathLM achieves competitive performance through domain-specific GRPO training. (a) Our 7B model (39.5%) approaches GPT-4o (40.4%) on WirelessMathBench-XL while using far fewer parameters than top performers DeepSeek-R1 and GPT-5 (>57%). (b) GRPO training from base models yields dramatic gains: doubling performance for 3B (+103%) and near-doubling for 7B (+81%), showing that verifiable rewards enable efficient domain specialization.

#### Contents

- 1 Introduction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 3
- 2 WirelessMathBench-XL: Dataset Construction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 3

- 2.1 Data Collection Pipeline . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 4
- 2.2 Mathematical Content Extraction and Problem Generation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 4
- 2.3 Quality Assurance Framework . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 5
- 2.4 Dataset Statistics and Analysis . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 5

- 3 Teaching Mathematical Reasoning with GRPO . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 6

- 3.1 Direct GRPO for Mathematical Reasoning . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 6
- 3.2 Verification-Based Reward System . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 6
- 3.3 Implementation Details . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 7

- 4 Experiments . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 7

- 4.1 Experimental Setup . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 7
- 4.2 Main Results on WirelessMathBench-XL . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 7
- 4.3 Generalization to General Mathematics . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 9
- 4.4 Qualitative Analysis . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 9

- 5 Related Work . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 10
- 6 Conclusion . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 10

- A Dataset Construction Details . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 16 A.1 Detailed Paper Collection Methodology . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 16
- B Quality Assessment Rubric For Human . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 17
- C Large Language Model-Assisted Quality Assessment . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 18

- C.1 Quality Assessment Framework . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 18
- C.2 Real LLM Annotation Examples . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 18

- D Prompt Construction for Dataset Generation and Evaluation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 19

- D.1 System Model Extraction Prompt . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 19
- D.2 Question Generation Prompt . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 20
- D.3 Quality Assessment Framework . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 21
- D.4 Standardized Evaluation Prompts . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 23

- E Representative System Model Extractions . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 24

- E.1 Example 1: Digital Twin-Assisted SIM-Based Air-Ground Communication . . . . . . . . . . . . . . . . . . . . . . . . . 24
- E.2 Example 2: Multi-UAV Patrol Inspection with Mobile Edge Computing . . . . . . . . . . . . . . . . . . . . . . . . . . 26
- E.3 Example 3: RIS-Aided Unsourced Random Access . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 28
- E.4 Model Extraction Quality Assessment . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 29

- F Human Expert Evaluation Examples . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 30

- F.1 Score 5 - Excellent Quality . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 30
- F.2 Score 4 - Good Quality . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 33
- F.3 Score 3 - Acceptable Quality . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 35
- F.4 Score 2 - Poor Quality . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 37
- F.5 Score 1 - Very Poor Quality . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 39

- G Representative Solution Examples from WirelessMathLM-7B . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 41

- G.1 High-Quality Solution Examples . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 41

- G.1.1 Multiple Choice Question: Matrix All-Pass Filter . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 41
- G.1.2 Fill-in-the-Blank (100%): Cell-Free Massive MIMO Beamforming . . . . . . . . . . . . . . . . . . . . . . . . . 42
- G.1.3 Fill-in-the-Blank (50%): Gaussian Function Components . . . . . . . . . . . . . . . . . . . . . . . . . . . . 43

- G.2 Error Analysis Examples . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 44

- G.2.1 Mathematical Equivalence Error . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 44
- G.2.2 Conceptual Misunderstanding Error . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 45
- G.2.3 MCQ Selection Error . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 46

#### 1 Introduction

Large language models (LLMs) demonstrate remarkable general reasoning capabilities [1, 8, 10, 14], yet they fail catastrophically when confronted with specialized technical mathematics [9, 16, 24, 26]. This limitation is particularly acute in wireless communications, where problems demand rigorous handling of convex optimization constraints, information-theoretic bounds, and complex-valued matrix algebra [24]. Consider determining optimal beamforming for multi-user wideband MIMO systems under power and interference constraints—a routine task in 5G/6G design that requires coordinating multiple mathematical frameworks.

The core challenge lies in a fundamental tension: achieving expert-level performance in specialized domains typically requires either massive scale or extensive domain-specific supervision, yet wireless systems demand computational efficiency and lack large-scale annotated datasets. While recent work has successfully adapted LLMs to specialized fields like medicine [31] and biology [4], these approaches rely on either abundant training data or expensive human feedback—resources that remain scarce in wireless communications. The recent WirelessMathBench [24] highlighted this gap with only 587 problems from 40 papers, far below the scale needed for robust model training.

We present WirelessMathLM, which resolves this tension through a key insight: technical mathematics possesses an inherent structure—verifiable correctness—that can substitute for both scale and supervision. Unlike open-ended tasks where quality assessment requires human judgment, wireless problems have deterministic correct answers that can be automatically verified. We exploit this property through Group Relative Policy Optimization (GRPO) [30], training compact models (0.5B–7B parameters) directly from base checkpoints using only binary verification signals.

As Figure 1 demonstrates, a 7B model trained with GRPO achieves 39.5% accuracy on wireless mathematics, approaching GPT-4o (40.4%) while using ≈100× fewer parameters than DeepSeek-R1 (671B, 57.4%). The improvements are consistent across scales—our 3B model doubles its accuracy (+103%), demonstrating that verification-based learning provides strong gradients even from sparse initial success. Most surprisingly, specialized training enhances rather than degrades general capabilities: our models gain an average of 8.4 points on standard mathematics benchmarks (MATH [17], Minerva-Math [22], OlympiadBench [16]) without any explicit training on these tasks.

To enable this approach, we construct WirelessMathBench-XL, creating 4,027 problems from 970 papers. Our three-tier problem design—multiple-choice for concept recognition, progressive fill-in-the-blank with 25%-75% masking for structured reasoning, and full equation completion for comprehensive mastery—provides both training signal and fine-grained evaluation. Each problem includes complete variable definitions and context, enabling automated verification of student responses while the dataset construction itself employs rigorous dual-layer quality assurance combining automated screening with expert validation.

Our contributions are threefold:

- • We demonstrate that verification alone enables efficient domain specialization. GRPO training from base models, without supervised warm-start or human feedback, consistently improves performance across all model scales (0.5B: +11%, 3B: +103%, 7B: +81%). This challenges the assumption that reinforcement learning requires extensive pre-training.
- • We show that specialized training develops transferable mathematical reasoning. The consistent gains on general benchmarks contradict conventional wisdom about catastrophic forgetting, suggesting that learning domain-specific mathematics strengthens fundamental capabilities.
- • We provide infrastructure for reproducible research. WirelessMathBench-XL, our trained models, and the GRPO training framework are publicly released to accelerate development of efficient, specialized AI for technical domains.

#### 2 WirelessMathBench-XL: Dataset Construction

Creating a high-quality benchmark for wireless communication mathematics requires addressing three key challenges: (1) extracting structured mathematical content from dense technical papers, (2) ensuring problem

[Figure 2]

[Figure 3]

[Figure 4]

Mathematical

Quality

[Figure 5]

[Figure 6]

[Figure 7]

###### Paper Collection

Extraction

Assurance

arXiv Multi-Category Crawling

DeepSeek-R1 Processing

Dual-Layer Validation

[Figure 8]

### 10-25

[Figure 9]

[Figure 10]

GPT-4o

Evaluation Auto

InitialCrawl ~47,000 GPT-4oFilter 3,186 FinalSelection 970

###### Formulas per Paper

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

6 Expert

###### Reviewers Manual

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

Quality Threshold

Fill-in

≥ 3/5

FEC

MCQ

###### 11,400

3,800

3,800

(25%-50%-75%)

eess.SP cs.IT

cs.NI

[Figure 21]

22% 78%

[Figure 22]

2005-2025

[Figure 23]

Figure 2 Overview of the WirelessMathBench-XL construction pipeline. Starting from 47,000 arXiv papers, GPT-4o filtering identifies 970 papers with substantial mathematical content. DeepSeek-R1 extracts 10-25 formulas per paper, generating multiple-choice questions, progressive fill-in-the-blank (25%-75% masking), and full equation completion problems. Quality assurance employs dual-layer screening: automated GPT-4o evaluation followed by expert validation, with 78% of questions meeting the quality threshold (score ≥ 3/5).

correctness and solvability, and (3) maintaining consistency across diverse mathematical formulations. We present a systematic pipeline that construct WirelessMathBench-XL from 970 papers, yielding 4,027 problems.

- Figure 2 illustrates our three-stage pipeline for constructing WirelessMathBench-XL from raw arXiv papers to validated mathematical questions.

##### 2.1 Data Collection Pipeline

We developed an automated pipeline that comprehensively collects and processes wireless communication papers from arXiv. Our approach prioritizes broad coverage with sophisticated filtering rather than narrow targeting.

Paper Collection and Filtering. We query 24 arXiv categories spanning core wireless domains (cs.NI, eess.SP, cs.IT), AI/ML (cs.LG, stat.ML), and interdisciplinary areas. Our crawler initially retrieves 47,000 papers from 2005-2025 using broad keyword queries across communication, signal processing, and networking terms. Each paper receives an automated relevance score based on keyword presence and category alignment. We then apply GPT-4-based filtering to identify 3,186 papers containing substantial mathematical content, from which we select the top ∼ 1,000 based on mathematical rigor, citation impact, and topical diversity. Full implementation details are provided in Appendix A.

##### 2.2 Mathematical Content Extraction and Problem Generation

Structured Model Extraction. We employ DeepSeek-R1 [14] to extract mathematical models from each paper’s LaTeX source. Our extraction preserves complete context including system equations, variable definitions with units and domain restrictions, underlying assumptions, and boundary conditions. Each paper yields a structured summary with properly formatted mathematical notations (e.g., v for vectors, H ∈ CN×M for complex matrices). Appendix E presents three representative examples of extracted system models demonstrating the comprehensiveness of our approach across different wireless domains: SIM-based airground communication, UAV-MEC systems, and RIS-aided random access.

Automated Problem Generation. From extracted models, we generate three types of exam-style questions using carefully designed prompt templates (see Appendix D for complete specifications):

- • Multiple Choice Questions (MCQ): Equations are presented with masked right-hand sides, accompanied by

- four carefully designed options. Distractors reflect common errors such as matrix dimension mismatches or incorrect operator sequences.
- • Progressive Fill-in-the-Blank (Fill-in): Four difficulty levels with 25%, 50%, and 75% of equation components masked, testing incremental understanding.
- • Full Equation Completion (FEC): Complete 100% masking requiring full equation recall

##### 2.3 Quality Assurance Framework

Automated Evaluation. Each generated question undergoes systematic evaluation by GPT-4o across four critical dimensions: mathematical correctness, variable completeness, answer verifiability, and pedagogical value. The evaluation employs a comprehensive 5-point quality rubric, which categorizes problems as invalid (score 1), poor (score 2), acceptable (score 3), good (score 4), or excellent (score 5). This automated screening utilizes specialized prompt templates described in Appendix D to ensure consistent evaluation criteria across all question types.

Expert Validation. Questions passing automated evaluation proceed to human expert review conducted by a team of six domain specialists comprising four PhD students and two postdoctoral researchers with expertise spanning optimization theory, information theory, signal processing, and network analysis. Each question undergoes independent evaluation by at least two experts who assess mathematical rigor, notational consistency, problem clarity, and relevance to wireless communications. Questions must achieve a minimum consensus score of 3/5 to qualify for dataset inclusion. The final acceptance rate of 78% reflects our stringent quality standards. Detailed scoring criteria and representative examples across all quality levels are provided in Appendices B and F.

##### 2.4 Dataset Statistics and Analysis

The WirelessMathBench-XL dataset comprises 4,027 problems derived from 970 papers, providing comprehensive coverage across wireless communications mathematics.

Technical Coverage. Figure 3 shows the distribution of mathematical techniques across source papers. Deep learning dominates (259 papers, 14.0%), followed by convex optimization (206, 11.2%) and MIMO/Massive MIMO (192, 10.4%). The dataset balances established techniques—beamforming (185), channel coding (115), federated learning (110)—with emerging paradigms including RIS/IRS (156), semantic communications (75), and NOMA (54). This distribution ensures representation of both foundational mathematics and frontier research areas.

Temporal Distribution. The dataset spans three technological generations: 3G/4G (2005-2018: 28 papers,

- 2.9%), 5G deployment (2019-2023: 317 papers, 32.7%), and 5G-Advanced/6G research (2024-2025: 625 papers, 64.4%). This temporal weighting toward recent work captures state-of-the-art techniques while maintaining theoretical foundations.

Question Format. All problems follow standardized structure with complete variable definitions including type specifications (scalar/vector/matrix), domain constraints (e.g., HRIS ∈ CM×N), and physical units. Mathematical notation remains uniform: boldface for vectors (v), bold capitals for matrices (H), and standard operators (diag, tr, ⊗). Fill-in-the-blank questions implement progressive difficulty through systematic masking (25%, 50%, 75%, 100%).

Quality Distribution. Expert evaluation reveals that 35.53% of questions achieve acceptable quality (score 3), 30.89% are rated good (score 4), and 11.08% reach excellence (score 5). Questions scoring below threshold (scores 1-2: 22.50%) undergo revision or exclusion.

Dataset Split. The 4,027 problems partition into training (3,227, 80%) and test (800, 20%) sets with balanced representation. Training set: Fill-in-75% (900), FEC (751), Fill-in-50% (680), MCQ (551), Fill-in-25% (345). Test set maintains proportional distribution: 218, 191, 160, 133, and 98 problems respectively.

Total: 970 papers

(14.0%)

250

259

(11.2%)

206

(10.4%)

(10.0%)

200

192

NumberofPapers

185

(8.5%)

156

150

(6.4%)

118

(5.5%)

101

100

(4.1%)

(4.0%)

(3.8%)

(3.7%)

75

(3.5%)

73

70

69

64

(2.5%)

(2.4%)

(2.3%)

(2.1%)

46

45

50

43

(1.7%)

39

(1.6%)

32

(1.2%)

(1.1%)

29

22

21

0

DeepLearningConvexOptimizationMIMO/MassiveMIMOBeamformingEdgeComputingRIS/IRSReinforcementLearningInformationTheory ISACOFDM/OFDMAUAVCommunicationsChannelCodingFederatedLearningChannelEstimationStochasticGeometryMachineLearningSemanticCommunicationsResourceAllocationNOMA CellFree

- Figure 3 Distribution of the top 20 key techniques across the 970 source papers in WirelessMathBench-XL. Deep learning leads with 259 papers (14.0%), followed by convex optimization (206, 11.2%) and MIMO/Massive MIMO (192, 10.4%). The distribution spans from foundational techniques (beamforming, channel coding) to emerging paradigms (RIS/IRS, semantic communications, NOMA)

#### 3 Teaching Mathematical Reasoning with GRPO

Teaching language models mathematical reasoning in specialized domains leverages a unique property: unlike general dialogue, wireless mathematics problems have verifiable correctness criteria. We employ GRPO [30] to directly train models from their base state, using automated verification as reward signals without expensive human feedback or supervised warm-start.

##### 3.1 Direct GRPO for Mathematical Reasoning

Given a base language model πθ and a wireless mathematics problem x, we aim to learn a policy that generates correct solutions y = (s1,...,sn,a) where si denotes reasoning steps and a is the final answer. Following Shao et al. [30], we optimize directly using the GRPO objective:

JGRPO(θ) = E x∼P(X)

{yi}Gi=1∼πθold(·|x)

G

1 G

min

i=1

πθ(yi|x) πθ

Ai,clip

(yi|x)

old

πθ(yi|x) πθ

,1 − ϵ,1 + ϵ Ai (1)

(yi|x)

old

where G = 8 responses are sampled per problem, ϵ = 0.2 for clipping, and the group-wise advantage is computed as:

ri − mean({rj}Gj=1) std({rj}Gj=1)

(2)

Ai =

This formulation provides learning signal even when success rates are low, as the model learns from relative comparisons within each problem group rather than absolute rewards.

- 3.2 Verification-Based Reward System Our reward system leverages the structured nature of wireless mathematics through multi-level verification:

r(x,y) = α · rformat(y) + (1 − α) · raccuracy(x,y) (3) where α = 0.1 balances format compliance with correctness. Format Reward (rformat): Ensures outputs follow expected structure with proper LaTeX formatting and \boxed{} final answers:

rformat(y) = L[regex match(y,".*boxed{.*}.*")] (4)

Accuracy Reward (raccuracy): Verifies correctness through a hierarchical evaluation system: (1) Direct matching: For multiple-choice questions, extract and compare letter answers. (2) Symbolic verification: For fill-in-theblank problems, normalize expressions (removing spaces, \mathbf, \boldsymbol) and check equivalence.

- 3.3 Implementation Details

We train WirelessMathLM models directly from Qwen2.5 base checkpoints (0.5B, 3B, 7B) [29] using GRPO without supervised warm-start. Training employs AdamW optimizer with learning rate 10−6, cosine annealing, and KL penalty β = 0.01. We train for 40 epochs (240 steps) with evaluation every 5 steps on the held-out test set. Generation uses temperature T = 0.6 for validation and T = 1.0 for training rollouts. Training utilizes 4 NVIDIA A6000 GPUs with training time scaling by model size: 0.5B (14 hours), 3B (40 hours), and 7B (61 hours). The reward function implements hierarchical verification combining format checking with answer validation as described in Section 3.2.

- 4 Experiments

- 4.1 Experimental Setup

Baselines. We benchmark WirelessMathLM against comprehensive baselines spanning proprietary and opensource models. Proprietary models include GPT-5 [27], GPT-4o [18], Claude-4.0-Sonnet [2], Gemini-2.5-Flash, and Gemini-2.5-Pro [11], representing state-of-the-art commercial systems. For open-source comparisons, we evaluate against general-purpose models including DeepSeek-R1 (671B) [14], DeepSeek-V3.1 (671B) [7], Llama-3.3-70B-Instruct [13], and Qwen2.5-72B-Instruct [35], as well as math-specialized models such as Qwen2.5-Math-72B-Instruct [36] and DeepSeekMath-7B-RL [30]. To isolate the impact of GRPO training, we include ablations using the corresponding Qwen2.5 base models (0.5B, 3B, 7B) without reinforcement learning.

Standardized Evaluation Protocol. To ensure fair comparison, all models receive identical prompts constructed from standardized templates (see Appendix D for complete specifications). Each prompt includes comprehensive variable definitions, equation context, and explicit formatting instructions. For MCQs, models must select from four options and provide their answer in \boxed{} format. Fill-in-the-blank problems demand all masked positions be correctly filled—partial solutions receive no credit. For complex expressions where simple matching fails, GPT-4.1-mini performs semantic equivalence checking under the same all-or-nothing criterion.

- 4.2 Main Results on WirelessMathBench-XL Table 1 presents comprehensive evaluation results on the WirelessMathBench-XL test set.

GRPO enables competitive performance with dramatic parameter reduction. Our 7B WirelessMathLM trained with GRPO achieves 39.5% overall accuracy, approaching the performance of GPT-4o (40.4%) while using orders of magnitude fewer parameters. This result is particularly striking when compared against opensource math-specialized models: our approach outperforms both Qwen2.5-Math-7B-Instruct (21.6%) and DeepSeekMath-7B-RL (21.5%) by nearly 2×, despite these models being explicitly trained for mathematical reasoning. The performance gain stems from our domain-specific training strategy—while general math models struggle with the specialized notation and problem structures in wireless communications, our targeted approach with verifiable rewards enables efficient learning of domain-specific patterns.

Table 1 Performance on WirelessMathBench-XL test set (800 problems). MCQ: Multiple Choice Questions, Fill-in: Fill-in-the-blank, FEC: Full Equation Completion. Best result per category in bold.

Model Size MCQ Fill-in FEC Overall

(%) (%) (%) (%) Proprietary Models

GPT-5 - 63.91 63.20 41.36 57.87 GPT-5-mini - 67.67 53.99 40.31 53.00 GPT-5-nano - 57.14 37.82 30.37 39.25 GPT-4o - 54.14 43.62 24.61 40.37 o4-mini - 67.67 49.56 40.31 50.38 Claude-4.0-Sonnet - 60.15 56.30 42.93 53.75 Gemini-2.5-Flash - 63.16 56.09 43.46 54.25 Gemini-2.5-Pro - 66.17 50.42 36.65 49.75 Grok-4-Fast - 70.31 56.33 40.33 54.89

Open-Source General Models

DeepSeek-R1 671B 65.41 60.50 43.98 57.37 DeepSeek-V3.1 671B 66.17 58.85 45.03 56.87 Llama-3.3-70B-Instruct 70B 54.14 38.03 28.27 38.37 Qwen2.5-72B-Instruct 72B 51.88 35.50 32.46 37.50 Qwen2.5-7B-Instruct 7B 39.1 21.85 26.18 25.75 Gemma 3 27B 27B 42.11 30.04 27.75 31.50 Gemma 3 12B 12B 36.84 21.43 21.99 24.12

Open-Source Math-Specialized Models

Qwen2.5-Math-72B-Instruct 72B 60.15 40.55 33.51 42.13 Qwen2.5-Math-7B-Instruct 7B 42.11 14.71 24.61 21.62 DeepSeekMath-7B-RL 7B 43.61 13.66 25.65 21.50

WirelessMathLM (Ours) Qwen2.5-7B-Base 7B 44.36 14.29 25.13 21.88

+ GRPO 7B 53.38 36.97 36.13 39.50 Qwen2.5-3B-Base 3B 26.32 7.14 15.71 12.37

+ GRPO 3B 48.87 17.02 28.80 25.12 Qwen2.5-0.5B-Base 0.5B 27.07 5.25 24.08 13.38

+ GRPO 0.5B 30.08 6.09 26.18 14.87

GRPO training yields consistent improvements across all model scales. The impact of GRPO training is substantial and scale-dependent. The 7B model nearly doubles its performance, improving from 21.9% to 39.5% (+81% relative), reaching within 0.9 percentage points of GPT-4o (40.4%). The 3B model demonstrates the most dramatic gains, more than doubling its accuracy from 12.4% to 25.1% (+103% relative). Even at minimal scale, the 0.5B model improves from 13.4% to 14.9% (+11% relative), suggesting that our dataset enables effective learning regardless of model capacity.

Performance patterns reveal task-specific strengths. Analyzing performance across question types reveals interesting patterns. All models perform best on multiple-choice questions (MCQ), where our 7B model achieves 53.4% accuracy—within striking distance of proprietary models like GPT-4o (54.1%) and approaching DeepSeek-R1 (65.4%). Performance on fill-in-the-blank questions shows the largest improvement from GRPO training (14.3% → 37.0% for 7B), suggesting that the reinforcement learning particularly helps with partial equation completion. Full equation completion (FEC) remains challenging across all models, though our 7B model’s 36.1% accuracy is competitive with GPT-5-mini (40.3%) and exceeds many larger open models.

Comparison with state-of-the-art reveals efficiency-performance trade-offs. While DeepSeek-R1 (671B) achieves the highest open-source performance at 57.4%, it requires ≈100× more parameters than our 7B model. The performance gap of 17.9 percentage points represents a favorable trade-off for deployment scenarios—our model achieves 69% of DeepSeek-R1’s performance with just 1% of its parameters. Among proprietary models, only GPT-5 (57.9%) significantly outperforms our approach, while models like Claude-4.0-Sonnet (53.8%)

and Gemini-2.5-Flash (54.3%) show more modest advantages despite their substantially larger scale and computational requirements.

- 4.3 Generalization to General Mathematics Surprisingly, training on wireless-specific mathematics enhances general mathematical reasoning (Table 2).

Domain-specific training strengthens fundamental mathematical capabilities. Our GRPO-trained models show substantial improvements on general mathematics benchmarks without any explicit training on these tasks. The 7B model improves from 52.0% to 67.0% on MATH 500 [17] (+28.8% relative), while the 3B model gains even more dramatically (41.6% → 58.2%, +39.9% relative). These improvements extend across diverse mathematical domains: Minerva-Math [21] sees modest but consistent gains (7B: 12.1% → 14.3%), OlympiadBench [16] improves substantially (7B: 25.3% → 30.2%), and AMC [23] performance increases significantly (7B: 27.7% → 41.0%). Even on the challenging AIME24 [23], the 7B model doubles its performance (6.7% → 13.3%).

Table 2 Transfer learning effects on general mathematical reasoning benchmarks.

Model MATH 500 Minerva-Math OlympiadBench AMC AIME24 Average

Qwen2.5-7B-Base 52.00 12.13 25.33 27.71 6.67 24.77 + GRPO 67.00 14.34 30.22 40.96 13.33 33.17

- ∆ (GRPO vs Base) +15.00 +2.21 +4.89 +13.25 +6.66 +8.40

Qwen2.5-3B-Base 41.60 5.88 14.67 18.07 0.00 16.04 + GRPO 58.20 9.93 22.96 21.69 0.00 22.56

- ∆ (GRPO vs Base) +16.60 +4.05 +8.29 +3.62 0.00 +6.52

- 4.4 Qualitative Analysis

To understand the reasoning capabilities developed through GRPO training, we conducted a comprehensive analysis of 800 solutions generated by WirelessMathLM-7B on WirelessMathBench-XL test problems spanning all quality levels (see Appendix F for representative examples).

Mathematical Reasoning Structure and Coherence. Our analysis reveals that WirelessMathLM-7B produces systematically structured solutions consistently. Across all evaluated problems, 99.1% of responses demonstrate clear step-by-step reasoning using logical connectives such as “therefore,” “thus,” and “hence.” The model exhibits great problem decomposition strategies. In complex scenarios involving multiple mathematical frameworks—such as MIMO beamforming under power constraints—solutions systematically establish physical principles before proceeding to mathematical derivations. For instance, when solving channel capacity problems, the model correctly identifies Shannon’s theorem applicability, establishes signal-to-noise ratio calculations, and methodically applies logarithmic transformations while maintaining dimensional consistency.

Domain-Specific Knowledge Integration. Analysis of correct solutions demonstrates strong competency in applying wireless-specific mathematical frameworks. Among correct responses, 87% properly identify the underlying problem type and select appropriate methodologies. This suggests successful integration of procedural knowledge (solution techniques) with conceptual understanding (physical principles). Consider the model’s approach to a Cell-Free Massive MIMO conjugate beamforming problem (Question ID 18369). The solution correctly identifies that conjugate beamforming requires complex conjugation of estimated channel coefficients, explains the physical rationale (“cancel out phase shifts introduced by the channel”), and derives the complete transmitted signal expression:

K

√ηmkgˆmk∗ uk (5)

sm = Pm

k=1

The response demonstrates understanding of power scaling, summation over users, and proper complex conjugation—all domain-specific requirements absent in general mathematical training.

Solution Quality Indicators and Mathematical Sophistication. Several qualitative indicators demonstrate that domain-specific GRPO training has developed genuine mathematical reasoning rather than pattern matching:

- (1)Constraint Awareness: The model consistently recognizes and applies physical constraints without explicit prompting. Solutions automatically incorporate non-negativity constraints for power allocations, maintain causality in signal processing derivations, and respect dimensionality requirements in matrix operations.
- (2)Method Justification: Correct solutions routinely include explicit rationales for chosen approaches. For example, in a matrix all-pass filter factorization problem (Question ID 11325), the model explains: “A matrix all-pass filter is a filter whose frequency response has a magnitude of 1 for all frequencies...” before deriving the G(z) = N(z)D−1(z) factorization and verifying the all-pass property through G(z)G−1(z) = Im.
- (3)Physical Intuition Integration: Solutions frequently connect mathematical expressions to underlying physical phenomena. When deriving XOR operations for backscattered data processing, the model explains the “commutative and associative” properties of XOR before applying them to wireless tag data recovery.

#### 5 Related Work

Mathematical Reasoning in LLMs. Chain-of-thought prompting [34] demonstrated that eliciting step-by-step reasoning significantly improves mathematical problem-solving in large language models. This was extended through process supervision [25], where models receive feedback on intermediate steps rather than just final answers, and tool-augmented approaches like ToRA [12] that integrate external computation for complex calculations. While these advances have been evaluated on benchmarks ranging from elementary word problems (GSM8K [6]) to competition mathematics (MATH [17]) and formal theorem proving (MiniF2F [37]), such benchmarks do not capture the symbolic manipulation and domain knowledge required in technical fields.

Domain Adaptation. Continued pre-training on domain-specific corpora [15] and instruction tuning [5] have proven effective for adapting language models to specialized fields. Scientific models like Galactica [32] attempted broad scientific reasoning, while BioBERT [20] and MedPaLM [31] achieved strong performance in biomedicine. Despite the mathematical intensity of wireless communications and its importance in 5G/6G systems, no prior work has developed specialized models for this domain.

Reinforcement Learning from Verifiable Rewards. While RLHF [28] successfully aligns language models with human preferences, it requires expensive annotation that limits scalability. Recent alternatives include Constitutional AI [3] using principle-based self-critique, RLAIF [19] leveraging model-generated feedback, and GRPO [30] using outcome-based rewards for mathematics.

#### 6 Conclusion

We demonstrated that verification-based reinforcement learning enables efficient domain specialization without massive scale or extensive supervision. Our key finding—that direct GRPO training from base models yields dramatic improvements (up to 103% for our 3B model) while enhancing rather than degrading general mathematical capabilities—challenges fundamental assumptions about both reinforcement learning prerequisites and catastrophic forgetting in domain adaptation. The success of WirelessMathLM, achieving near-GPT-4o performance with only 7B parameters, suggests that technical domains possessing verifiable correctness criteria constitute a distinct class of problems where compact, specialized models can match or exceed much larger general-purpose systems. This principle extends beyond wireless communications to any field with formal verification—circuit design, control theory, cryptography—where our approach of exploiting domain structure through binary verification rewards could replace expensive annotation or massive scale. By releasing WirelessMathBench-XL, our trained models, and the training codes, we provide concrete tools for the research community to explore this efficiency-through-verification paradigm, potentially transforming how specialized AI systems are developed for technical domains where correctness is paramount and computational resources are constrained.

#### Ethics Statement

We adhere to the ICLR Code of Ethics. This work focuses on advancing the mathematical reasoning capabilities of language models in the specialized domain of wireless communications. The WirelessMathBench-XL dataset was constructed from publicly accessible academic papers on arXiv, respecting the norms of scientific dissemination. Our data collection process did not involve human subjects or personally identifiable information. The expert validation phase was conducted by graduate students and postdoctoral researchers as part of their standard research activities. While any powerful AI technology carries potential for misuse, our work is foundational and does not present immediate dual-use concerns. We acknowledge that our dataset, being derived from existing literature, may reflect the inherent biases present in the field. We encourage responsible use of our models and dataset, and we are committed to addressing any ethical concerns that may arise.

#### Reproducibility Statement

To facilitate reproducibility of our work, we provide comprehensive details of our experimental methodology and make key resources publicly available. The complete WirelessMathBench-XL dataset containing 4,027 problems and evaluation results from all tested models is currently accessible at https://lixin.ai/ WirelessMathLM. Our dataset construction pipeline is thoroughly documented in Section 2, with detailed prompt templates, quality rubrics, and extraction procedures provided in Appendices D through E. The GRPO training methodology is fully specified in Section 3, including the complete mathematical formulation of our reward system (Equations 1-4), hyperparameter settings, and implementation details. Our experimental protocol described in Section 4 provides exact evaluation procedures, model configurations, and standardized prompt templates used for all baseline comparisons. The appendices contain extensive documentation including representative problem examples across all quality levels (Appendix F), detailed solution analyses from our models (Appendix G), and comprehensive error taxonomies that enable understanding of model behavior. Upon paper acceptance, we will release the complete codebase including the GRPO training framework, all model checkpoints (0.5B, 3B, and 7B parameters), and evaluation scripts to ensure full reproducibility of our results. All experiments were conducted on NVIDIA A6000 GPUs with computational requirements documented in Section 3, enabling researchers to estimate resources needed for replication.

#### References

- [1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.

- [2] Anthropic. Claude 4 sonnet. https://www.anthropic.com/claude/sonnet, 2025.
- [3] Yuntao Bai, Saurav Kadavath, Sandipan Kundu, Amanda Askell, Jackson Kernion, Andy Jones, Anna Chen, Anna Goldie, Azalia Mirhoseini, Cameron McKinnon, Carol Chen, Catherine Olsson, Christopher Olah, Danny Hernandez, Dawn Drain, Deep Ganguli, Dustin Li, Eli Tran-Johnson, Ethan Perez, Jamie Kerr, Jared Mueller, Jeffrey Ladish, Joshua Landau, Kamal Ndousse, Kamile Lukosuite, Liane Lovitt, Michael Sellitto, Nelson Elhage, Nicholas Schiefer, Noemi Mercado, Nova DasSarma, Robert Lasenby, Robin Larson, Sam Ringer, Scott Johnston, Shauna Kravec, Sheer El Showk, Stanislav Fort, Tamera Lanham, Timothy Telleen-Lawton, Tom Conerly, Tom Henighan, Tristan Hume, Samuel R. Bowman, Zac Hatfield-Dodds, Ben Mann, Dario Amodei, Nicholas Joseph, Sam McCandlish, Tom Brown, and Jared Kaplan. Constitutional ai: Harmlessness from ai feedback. ArXiv, abs/2212.08073, 2022. URL https://api.semanticscholar.org/CorpusID:254823489.

- [4] Zeming Chen, Alejandro Hern’andez Cano, Angelika Romanou, Antoine Bonnet, Kyle Matoba, Francesco Salvi, Matteo Pagliardini, Simin Fan, Andreas Kopf, Amirkeivan Mohtashami, Alexandre Sallinen, Alireza Sakhaeirad, Vinitra Swamy, Igor Krawczuk, Deniz Bayazit, Axel Marmet, Syrielle Montariol, Mary-Anne Hartley, Martin Jaggi, and Antoine Bosselut. Meditron-70b: Scaling medical pretraining for large language models. ArXiv, abs/2311.16079,

2023. URL https://api.semanticscholar.org/CorpusID:265456229.

- [5] Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Yunxuan Li, Xuezhi Wang, Mostafa Dehghani, Siddhartha Brahma, Albert Webson, Shixiang Shane Gu, Zhuyun Dai, Mirac Suzgun, Xinyun Chen, Aakanksha Chowdhery, Alex Castro-Ros, Marie Pellat, Kevin Robinson, Dasha Valter, Sharan Narang, Gaurav Mishra,

- Adams Yu, Vincent Zhao, Yanping Huang, Andrew Dai, Hongkun Yu, Slav Petrov, Ed H. Chi, Jeff Dean, Jacob Devlin, Adam Roberts, Denny Zhou, Quoc V. Le, and Jason Wei. Scaling instruction-finetuned language models, 2022. URL https://arxiv.org/abs/2210.11416.
- [6] Karl Cobbe, Vineet Kosaraju, Mo Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. Training verifiers to solve math word problems. ArXiv, abs/2110.14168, 2021. URL https://api.semanticscholar.org/CorpusID:239998651.

- [7] DeepSeek-AI. Deepseek-v3.1 model introduction. https://www.deepseek.com/, 2025. Accessed: 2025-09-21.
- [8] Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.

- [9] Simon Frieder, Luca Pinchetti, Ryan-Rhys Griffiths, Tommaso Salvatori, Thomas Lukasiewicz, Philipp Petersen, and Julius Berner. Mathematical capabilities of chatgpt. Advances in neural information processing systems, 36, 2024.

- [10] Google DeepMind. Introducing gemini 2.0: our new ai model for the agentic era, 2024. URL https://blog.google/ technology/google-deepmind/google-gemini-ai-update-december-2024/. Accessed: 2024-12-11.
- [11] Google DeepMind. Gemini 2.5 models. https://deepmind.google/technologies/gemini/, 2025.
- [12] Zhibin Gou, Zhihong Shao, Yeyun Gong, Yelong Shen, Yujiu Yang, Minlie Huang, Nan Duan, and Weizhu Chen. Tora: A tool-integrated reasoning agent for mathematical problem solving. ArXiv, abs/2309.17452, 2023. URL https://api.semanticscholar.org/CorpusID:263310365.

- [13] Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, Amy Yang, Angela Fan, and et al. The llama 3 herd of models,

2024. URL https://arxiv.org/abs/2407.21783.

- [14] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.

- [15] Suchin Gururangan, Ana Marasovi´c, Swabha Swayamdipta, Kyle Lo, Iz Beltagy, Doug Downey, and Noah A. Smith. Don’t stop pretraining: Adapt language models to domains and tasks. ArXiv, abs/2004.10964, 2020. URL https://api.semanticscholar.org/CorpusID:216080466.

- [16] Chaoqun He, Renjie Luo, Yuzhuo Bai, Shengding Hu, Zhen Thai, Junhao Shen, Jinyi Hu, Xu Han, Yujie Huang, Yuxiang Zhang, Jie Liu, Lei Qi, Zhiyuan Liu, and Maosong Sun. OlympiadBench: A challenging benchmark for promoting AGI with olympiad-level bilingual multimodal scientific problems. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar (eds.), Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 3828–3850, Bangkok, Thailand, August 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.acl-long.211. URL https://aclanthology.org/2024.acl-long.211/.

- [17] Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset. arXiv preprint arXiv:2103.03874, 2021.

- [18] Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card. arXiv preprint arXiv:2410.21276, 2024.

- [19] Harrison Lee, Samrat Phatale, Hassan Mansoor, Kellie Lu, Thomas Mesnard, Colton Bishop, Victor Carbune, and Abhinav Rastogi. Rlaif vs. rlhf: Scaling reinforcement learning from human feedback with ai feedback. In International Conference on Machine Learning, 2023. URL https://api.semanticscholar.org/CorpusID: 261493811.

- [20] Jinhyuk Lee, WonJin Yoon, Sungdong Kim, Donghyeon Kim, Sunkyu Kim, Chan Ho So, and Jaewoo Kang. Biobert: a pre-trained biomedical language representation model for biomedical text mining. Bioinformatics, 36:1234 – 1240,

2019. URL https://api.semanticscholar.org/CorpusID:59291975.

- [21] Aitor Lewkowycz, Anders Andreassen, David Dohan, Ethan Dyer, Henryk Michalewski, Vinay Ramasesh, Ambrose Slone, Cem Anil, Imanol Schlag, Theo Gutman-Solo, et al. Solving quantitative reasoning problems with language models. Advances in Neural Information Processing Systems, 35:3843–3857, 2022.

- [22] Aitor Lewkowycz, Anders Johan Andreassen, David Dohan, Ethan Dyer, Henryk Michalewski, Vinay Venkatesh Ramasesh, Ambrose Slone, Cem Anil, Imanol Schlag, Theo Gutman-Solo, Yuhuai Wu, Behnam Neyshabur, Guy Gur-Ari, and Vedant Misra. Solving quantitative reasoning problems with language models. In Alice H. Oh, Alekh Agarwal, Danielle Belgrave, and Kyunghyun Cho (eds.), Advances in Neural Information Processing Systems, 2022. URL https://openreview.net/forum?id=IFXTZERXdM7.

- [23] Jia Li, Edward Beeching, Lewis Tunstall, Ben Lipkin, Roman Soletskyi, Shengyi Huang, Kashif Rasul, Longhui Yu, Albert Q Jiang, Ziju Shen, et al. Numinamath: The largest public dataset in ai4maths with 860k pairs of competition math problems and solutions. Hugging Face repository, 13(9):9, 2024.

- [24] Xin Li, Mengbing Liu, Li Wei, Jiancheng An, M´erouane Debbah, and Chau Yuen. WirelessMathBench: A Mathematical Modeling Benchmark for LLMs in Wireless Communications. In Findings of the Association for Computational Linguistics: ACL 2025, 2025.

- [25] Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harrison Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net/forum?id=v8L0pN6EOi.

- [26] Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. arXiv preprint arXiv:2310.02255, 2023.

- [27] OpenAI. Introducing gpt-5. https://openai.com/index/introducing-gpt-5/, 2025.
- [28] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35:27730–27744, 2022.

- [29] Qwen, :, An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin, Kai Dang, Keming Lu, Keqin Bao, Kexin Yang, Le Yu, Mei Li, Mingfeng Xue, Pei Zhang, Qin Zhu, Rui Men, Runji Lin, Tianhao Li, Tianyi Tang, Tingyu Xia, Xingzhang Ren, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yu Wan, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zihan Qiu. Qwen2.5 technical report, 2025. URL https://arxiv.org/abs/2412.15115.
- [30] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Y Wu, et al. DeepSeekMath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

- [31] Karan Singhal, Shekoofeh Azizi, Tao Tu, S. Sara Mahdavi, Jason Wei, Hyung Won Chung, Nathan Scales, Ajay Tanwani, Heather Cole-Lewis, Stephen Pfohl, Perry Payne, Martin Seneviratne, Paul Gamble, Chris Kelly, Nathaneal Scharli, Aakanksha Chowdhery, Philip Mansfield, Blaise Aguera y Arcas, Dale Webster, Greg S. Corrado, Yossi Matias, Katherine Chou, Juraj Gottweis, Nenad Tomasev, Yun Liu, Alvin Rajkomar, Joelle Barral, Christopher Semturs, Alan Karthikesalingam, and Vivek Natarajan. Large language models encode clinical knowledge. Nature, 620:172 – 180,

2022. URL https://api.semanticscholar.org/CorpusID:255124952.

- [32] Ross Taylor, Marcin Kardas, Guillem Cucurull, Thomas Scialom, Anthony S. Hartshorn, Elvis Saravia, Andrew Poulton, Viktor Kerkez, and Robert Stojnic. Galactica: A large language model for science. ArXiv, abs/2211.09085, 2022. URL https://api.semanticscholar.org/CorpusID:253553203.

- [33] Bin Wang, Chao Xu, Xiaomeng Zhao, Linke Ouyang, Fan Wu, Zhiyuan Zhao, Rui Xu, Kaiwen Liu, Yuan Qu, Fukai Shang, Bo Zhang, Liqun Wei, Zhihao Sui, Wei Li, Botian Shi, Yu Qiao, Dahua Lin, and Conghui He. Mineru: An open-source solution for precise document content extraction, 2024. URL https://arxiv.org/abs/2409.18839.
- [34] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-ofthought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824–24837, 2022.

- [35] An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin, Kai Dang, Keming Lu, Keqin Bao, Kexin Yang, Le Yu, Mei Li, Mingfeng Xue, Pei Zhang, Qin Zhu, Rui Men, Runji Lin, Tianhao Li, Tingyu Xia, Xingzhang Ren, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yu Wan, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zihan Qiu. Qwen2.5 technical report. arXiv preprint arXiv:2412.15115, 2024.

- [36] An Yang, Beichen Zhang, Binyuan Hui, Bofei Gao, Bowen Yu, Chengpeng Li, Dayiheng Liu, Jianhong Tu, Jingren Zhou, Junyang Lin, et al. Qwen2. 5-math technical report: Toward mathematical expert model via self-improvement. arXiv preprint arXiv:2409.12122, 2024.

- [37] Kunhao Zheng, Jesse Michael Han, and Stanislas Polu. Minif2f: a cross-system benchmark for formal olympiad-level mathematics. ArXiv, abs/2109.00110, 2021. URL https://api.semanticscholar.org/CorpusID:237372712.

#### Use of Large Language Models

In accordance with the ICLR 2026 policy on Large Language Model (LLM) usage, we disclose that LLMs were utilized as tools in various stages of this research project. The final responsibility for all content, including its accuracy and originality, rests with the human authors.

- • Writing and Editing: LLMs were used to assist with improving the grammar, clarity, and style of the manuscript. The authors reviewed and edited all LLM-generated text to ensure it accurately reflects our research and findings.
- • Literature Discovery: LLMs were employed to help summarize related work and accelerate the literature discovery process, assisting in identifying relevant prior research in mathematical reasoning and domain adaptation.
- • Dataset Curation Pipeline: As detailed in Section C, LLMs were integral to the construction of the WirelessMathBench-XL dataset. Specifically:

- – Paper Filtering: GPT-4o was used to perform an initial filtering of ∼47,000 papers to identify those with substantial mathematical content relevant to wireless communications.
- – Content Extraction: DeepSeek-R1 was used to extract structured mathematical models from the LaTeX source of selected papers.
- – Automated Quality Assessment: GPT-4o were used as part of a multi-tier quality assurance framework to perform initial automated evaluations of generated questions.

- • Evaluation: For our evaluation metric, GPT-4.1-mini was used to perform semantic equivalence checking on complex mathematical expressions where simple string matching was insufficient.

In all instances, LLM outputs were critically reviewed, validated, and verified by the authors. We take full responsibility for the claims, results, and conclusions presented in this paper.

- A Dataset Construction Details A.1 Detailed Paper Collection Methodology Multi-Category Coverage. We query across 24 arXiv categories to capture interdisciplinary research:

- • Core categories: cs.NI (Networking), eess.SP (Signal Processing), cs.IT (Information Theory)
- • AI/ML categories: cs.LG, stat.ML, cs.AI for learning-based approaches
- • Systems categories: cs.SY, cs.DC, cs.MA for distributed and multi-agent systems
- • Physics categories: physics.optics, quant-ph for emerging physical layer techniques
- • Mathematical categories: math.OC, math.IT for optimization and theory

Query Construction Strategy. We implement four complementary query strategies: queries = [

{name: ’basic_communication_terms’, keywords: [communication, network, wireless, radio, signal,

antenna, frequency, spectrum, transmission]}, {name: ’system_algorithm_terms’,

keywords: [system, algorithm, optimization, performance,

model, framework, architecture]}, {name: ’application_computing_terms’,

keywords: [computing, sensing, iot, edge, cloud,

distributed, energy, security]}, {name: ’data_intelligence_terms’,

keywords: [learning, intelligence, neural, prediction, detection, processing, estimation]}

] Relevance Scoring and Annotation. For each paper, we calculate:

- • Relevance Score (0-1): Weighted sum of keyword presence in title (0.6 weight) and abstract (0.3 weight), plus category bonuses (eess.SP: 0.4, cs.NI: 0.35, cs.IT: 0.3)
- • Technology Focus: Detected across 8 categories (wireless basic, advanced wireless, next gen, emerging tech, signal processing, network protocol, ai ml, iot apps)

- • Quality Tier: Based on relevance score (high: ≥0.7, medium: 0.4-0.7, low: 0.1-0.4)
- • Research Type: Classified as survey, algorithmic, analytical, experimental, or theoretical

PDF Processing. Papers undergo full-text processing using:

- • MinerU [33] for PDF-to-markdown conversion preserving LaTeX equations
- • Batch processing of 3-5 PDFs concurrently ( 40 seconds per paper)
- • Rate-limited arXiv bulk access API with 3-second delays

#### B Quality Assessment Rubric For Human

Table 3 Detailed expert question quality assessment rubric

Score Criteria

- 1 - Invalid Problem statement or solution is clearly wrong or contradictory; Not related to wireless communications domain; Cannot be used as a valid question

- 2 - Poor Statement correct but problem too trivial (answerable instantly); Problem too vague or nearly impossible to answer correctly; Very little learning or evaluation value

- 3 - Acceptable Statement and solution reasonable with no major errors; Difficulty and relevance are average; Can be kept but adds limited value (baseline quality)

- 4 - Good Clear and well-structured problem; Relevant to domain and moderately challenging; Provides meaningful assessment of understanding; Worth keeping and recommending

- 5 - Excellent Highly relevant to the domain; Strong depth, creativity, or insight required; Excellent for differentiating levels of understanding; Strongly recommended for inclusion

#### C Large Language Model-Assisted Quality Assessment

This section presents our comprehensive approach to leveraging large language models (LLMs) for scalable quality assessment of mathematical questions in wireless communications. Our methodology addresses the fundamental challenge of maintaining expert-level evaluation standards while achieving the scale necessary for large dataset curation.

Role in the Overall Annotation Pipeline: This LLM-assisted quality assessment serves as the first filtering stage in our comprehensive annotation pipeline for our method. The complete pipeline consists of two sequential stages: (1) LLM-based filtering using our enhanced prompt system to automatically identify and remove low-quality questions, reducing the workload for human annotators; (2) Expert human annotation where domain experts review the filtered questions and provide detailed quality assessments;

##### C.1 Quality Assessment Framework

Our quality assessment framework employs a systematic approach to evaluate technical questions across six dimensions:

- 1. Question Clarity (1-5): Measures the clarity and unambiguousness of the question statement
- 2. Background Relevance (1-5): Evaluates the completeness and relevance of provided context
- 3. Answer Accuracy (1-5): Assesses the correctness and formatting of the provided answer
- 4. Technical Appropriateness (1-5): Determines if the difficulty level matches the target audience
- 5. Mathematical Rigor (1-5): Evaluates mathematical notation and conventions
- 6. Wireless Communication Relevance (1-5): Measures domain relevance to wireless communications

##### C.2 Real LLM Annotation Examples

To demonstrate the practical effectiveness of our LLM-assisted quality assessment system, we present three representative examples from our evaluation dataset, showcasing different quality levels and the corresponding LLM assessments.

Table 4 LLM Annotation Examples Across Quality Levels

Quality Level Question Content LLM Assessment High Quality Background: Federated fine-tuning system

LLM Score: 4/5 Strengths: Clear structure, complete context, accurate answer, rigorous notation Weaknesses: Could benefit from brief explanation of low-rank adaptation significance Agreement: ✓

with low-rank adaptation matrices Ak ∈ Rd×r, Bk ∈ Rr×d Question: Which term completes: W + [MASK]?

Options: A) AkBk B) BkAk Human: 4/5

LLM Score: 3/5 Strengths: Wireless relevance, accurate answer Weaknesses: Ambiguous [MASK] usage, lacks clarity in instructions Technical Issues: Missing variable definitions Disagreement: LLM too optimistic

Medium Quality Background: H-NOMA system with variable definitions partially provided Question: Fill in [MASK] for the equation Human: 2/5

Low Quality Background: Transformer model context with incomplete variable definitions Question: What replaces the ”full key matrix”? Human: 1/5

LLM Score: 3/5 Strengths: Clear structure, accurate answer Weaknesses: Limited wireless relevance, focuses more on tensor parallelism Technical Issues: ”Full key matrix” not defined Bias: LLM shows optimistic scoring pattern

#### D Prompt Construction for Dataset Generation and Evaluation

We employ specialized prompt templates for dataset construction, quality assessment, and standardized evaluation to ensure consistency and fairness across all stages of our methodology.

- D.1 System Model Extraction Prompt The following prompt template guides the extraction of mathematical models from research papers:

System Model Extraction Template

Task: Act as an expert in wireless communications and mathematical modeling. Extract and summarize the mathematical system modeling from the paper.

<<STRUCTURE REQUIREMENTS>>

- 1. **Model Extraction**:

- a) Identify ALL system equations with context
- b) For each equation:

- i) List ALL variables with units/dimensions
- ii) Specify underlying assumptions
- iii) Note domain restrictions

- 2. **Summary Organization**: \paragraph{Background} (2-3 sentences contextualizing the model) \paragraph{Key Assumptions} (bullet points with $\bullet$) \paragraph{Parameter Definitions} (table-like structure) \paragraph{Core Equations} (numbered with original labels)
- 3. **Equation Formatting**:

- - Vectors: \boldsymbol{v}
- - Matrices: \mathbf{M}
- - Operators: \mathrm{diag}, \mathrm{tr}
- - Complex numbers: j for imaginary unit

<<CONTENT GUIDELINES>>

- - Variable Explanations:
- - For each symbol: $\theta$ (Type: Phase shift; Domain: [0,2 $\pi$); Unit: rad)
- - Matrix dimensions: $\mathbf{H} \in \mathbb{C}^{N\times M}$
- - Distinguish similar symbols: $h_{ij}$ vs $h_{i}^{(j)}$
- - Model Validation:
- - Verify dimensional consistency
- - Check boundary conditions
- - Confirm parameter unit homogeneity

- D.2 Question Generation Prompt The following template generates exam-style questions from extracted models:

Question Generation Template

Task: Generate exam-style questions from research paper summaries. <<STRUCTURE REQUIREMENTS>>

- 1. **Per Equation Processing**:

- a) Identify ALL system model equations
- b) For EACH equation:

- i) Mask the RHS with [MASK]
- ii) Generate 1 MCQ with 4 plausible options
- iii) Create 4 progressive fill-in-the-blank subquestions:

- 25%, 50%, 75%, and 100% key symbols masked

- 2. **Question Components**:

- For MCQs:

- * Background: MUST include detailed variable definitions Format: "where $\boldsymbol{x}$ is the transmitted signal vector, $\mathbf{H} \in \mathbb{C}^{N \times M}$ represents the channel matrix..."
- * Equation: Masked equation in display math mode
- * Question: Explicitly ask to replace [MASK]
- * Options: 4 LaTeX-formatted choices (A)-(D)
- * Answer: Detailed derivation walkthrough

<<ENHANCED BACKGROUND REQUIREMENTS>>

- - Variable Definition Format:
- - Start with system context: "In this [type of system]..."
- - List EVERY symbol that appears in the equation
- - Include matrix/vector dimensions
- - Specify units where applicable: "(in watts)", "(in Hz)"
- - Explain subscripts and superscripts
- - Distractor Design:

- 1) Matrix dimension mismatches
- 2) Incorrect operator sequences
- 3) Missing diag() operators
- 4) Channel matrix transposition errors
- 5) Incorrect matrix multiplication order

- - Masking Strategy:
- - 25%: Single critical variable
- - 50%: Two interdependent terms
- - 75%: Multiple components
- - 100%: Full equation recall

##### D.3 Quality Assessment Framework

To ensure consistent quality evaluation across the dataset, we employ a comprehensive assessment framework with few-shot learning enhancement. This framework guides both automated LLM evaluation and human expert review.

Quality Assessment Prompt with Few-Shot Learning

You are an expert evaluator specializing in wireless communication and mathematics education. Your task is to assess the quality of technical questions designed for advanced undergraduate and graduate students in wireless communications.

## EVALUATION METHODOLOGY Follow this systematic approach:

- ### STEP 1: Initial Question Analysis

- - Read the question, background, equation, and answer carefully
- - Identify the technical domain and complexity level
- - Check for obvious errors or inconsistencies

- ### STEP 2: Multi-Dimensional Quality Assessment Evaluate each dimension on a 1-5 scale:

- 1. Question Clarity (1-5): Crystal clear vs confusing/incomprehensible
- 2. Background Relevance (1-5): Comprehensive context vs inadequate background
- 3. Answer Accuracy (1-5): Completely correct vs incorrect/flawed
- 4. Technical Appropriateness (1-5): Perfect difficulty vs inappropriate level
- 5. Mathematical Rigor (1-5): Excellent notation vs poor rigor
- 6. Wireless Relevance (1-5): Highly relevant vs not relevant

## HUMAN EXPERT EXAMPLES Learn from these actual expert evaluations:

- Example 1 - Score: 1 (Very Poor) Question: "Which expression correctly calculates the sensitivity metric?" Human Feedback: "The definition of TN is not given"

→ Missing variable definitions make question unsolvable

- Example 2 - Score: 3 (Acceptable) Question: "Which performance metric should replace [MASK]?" Human Feedback: "Some variables in choices are not given"

→ Minor gaps but workable with assumptions

- Example 3 - Score: 5 (Excellent) Question: Complete differential privacy equation with full context Human Feedback: "Well-structured with complete information"

→ Ready for immediate use

## CRITICAL EVALUATION GUIDELINES Be especially strict about:

- - Missing Variable Definitions: Any undefined variables→Score$\leq$ 2
- - Incomplete Context: Key background missing → Score $\leq$ 2
- - Vague Problem Statements: Ambiguous questions → Score $\leq$ 3
- - Technical Accuracy: Mathematical/technical errors → Score $\leq$ 2

## OUTPUT FORMAT Provide assessment in JSON: {

"overall_score": [1-5 integer], "dimension_scores": {

"question_clarity": [1-5],

"background_relevance": [1-5], "answer_accuracy": [1-5], "technical_appropriateness": [1-5], "mathematical_rigor": [1-5], "wireless_relevance": [1-5]

}, "binary_flags": {

"is_correct": [true/false], "is_wireless_related": [true/false]

}, "quality_analysis": {

"strengths": ["Key strengths"], "weaknesses": ["Areas for improvement"], "specific_improvements": ["Detailed suggestions"]

} }

Question Type: {question_type} Question Text: {question_text} Background: {background} Equation: {equation} Options: {options} Correct Answer: {correct_answer}

##### D.4 Standardized Evaluation Prompts

To ensure reproducible evaluation, all models receive identical prompts constructed from the following templates:

MCQ Evaluation Template:

MCQ Evaluation Template

**Background** [Complete variable definitions and system context]

**Question** [Question text]

**Equation** [Equation with [MASK] placeholder]

**Options**

- A: [Option A]
- B: [Option B]
- C: [Option C]
- D: [Option D]

--Please analyze this problem step by step. Show your reasoning and calculations. Your final answer should be given at the end in the format: \boxed{X} where X is the letter of the correct option.

###### Fill-in-the-Blank Evaluation Template:

Fill-in-the-Blank Evaluation Template

**Background** [Complete variable definitions and system context]

**Question** [Question text]

**Equation** [Equation with [MASK] placeholder(s)]

--Please solve this problem step by step. Fill in the [MASK] placeholder(s) with the correct mathematical expression(s).

For single mask: Your final answer should be given at the end in the format: \boxed{your\_answer}

For multiple masks: Your final answers should be given at the end in the format: \boxed{answer1}, \boxed{answer2}, ... (for the N blanks in order)

#### E Representative System Model Extractions

This section presents three representative examples of system models extracted by DeepSeek-R1 from research papers in our corpus. These examples demonstrate the diversity and complexity of mathematical formulations captured in WirelessMathBench-XL.

##### E.1 Example 1: Digital Twin-Assisted SIM-Based Air-Ground Communication

This model integrates multi-layer stacked intelligent metasurface (SIM) beamforming with eVTOL trajectory optimization, representing the convergence of aerial communications and reconfigurable surface technologies.

SIM-Based Air-Ground Communication System

Background This paper proposes a Digital Twin (DT)-assisted framework for joint optimization of Stacked Intelligent Metasurface (SIM)-based air-ground communication and electric Vertical Take-off and Landing (eVTOL) flight control within prescribed air corridors. The system model integrates a multi-layer SIM beamforming structure at the Air Traffic Control (ATCo) station with a composite potential field method for eVTOL trajectory planning, aiming to maximize the sum transmission rate while ensuring safe navigation.

Key Assumptions • The air-ground channel between the SIM and each eVTOL follows a Rician fading model.

- • Each meta-atom on the SIM imposes an ideal, configurable phase shift without amplitude attenuation.
- • The transmission matrices Wl between metasurface layers are modeled based on Rayleigh-Sommerfeld diffraction theory, assuming perfect knowledge of the SIM’s physical structure.
- • eVTOLs fly within a predefined, non-overlapping air corridor Rcor.
- • The signals for different eVTOLs are independent and identically distributed (i.i.d.) with zero mean and unit variance.
- • The additive receiver noise is independent, circularly symmetric complex Gaussian (AWGN). Parameter Definitions B = [xATC, yATC, 0]T (Type: ATCo station position; Domain: R3×1; Unit: m)

- M (Type: Number of eVTOLs / ATCo antennas; Domain: Z+; Unit: None) L (Type: Number of metasurface layers in SIM; Domain: Z+; Unit: None) K (Type: Number of meta-atoms per metasurface layer; Domain: Z+; Unit: None)
- N (Type: Number of discrete time slots; Domain: Z+; Unit: None) δ (Type: Duration of a time slot; Domain: R+; Unit: s)

qm[n] = [xeVm TOL[n], ymeV TOL[n], zmeV TOL[n]]T (Type: 3D position of eVTOL m at time n; Domain: Rcor ⊂ R3; Unit: m) Vmax (Type: Maximum eVTOL velocity; Domain: R+; Unit: m/s) PATC (Type: Total available transmission power at ATCo; Domain: R+; Unit: W) pm[n] (Type: Transmission power allocated to eVTOL m at time n; Domain: R+; Unit: W) θkl [n] (Type: Phase shift of meta-atom k on layer l at time n; Domain: [0, 2π); Unit: rad) Ψl[n] = diag(ejθ1l[n], ..., ejθKl [n]) (Type: Phase shift matrix for layer l at time n; Domain: CK×K; Unit: None) Wl (Type: Transmission matrix between layers l − 1 and l; Domain: CK×K; Unit: None) wm1 (Type: Transmission vector from ATCo antenna m to first metasurface layer; Domain: CK×1; Unit: None) λ (Type: Carrier wavelength; Domain: R+; Unit: m) dx, dy (Type: Size of a meta-atom along x and y axes; Domain: R+; Unit: m) hHm[n] (Type: Channel vector from last SIM layer to eVTOL m at time n; Domain: C1×K; Unit: None) ρ0 (Type: Reference path loss at 1m; Domain: R+; Unit: None (often in dB)) αh (Type: Path loss exponent; Domain: R≥2; Unit: None) κh (Type: Rician factor; Domain: R+; Unit: dB) σm2 (Type: Receiver noise power at eVTOL m; Domain: R+; Unit: W) sm[n] (Type: Transmission data symbol for eVTOL m at time n; Domain: C; Unit: None; Assumption: E{sm[n]} = 0, E{|sm[n]|2} = 1, i.i.d.) Core Equations 1) SIM Beamforming Matrix. The end-to-end beamforming matrix G[n] of the L-layer SIM is given by the product of the transmission and phase shift matrices across all layers.

G[n] = ΨL[n]WLΨL−1[n] · · · Ψ2[n]W2Ψ1[n] ∈ CK×K

- 2) Inter-layer Transmission Matrix Entry. The (k, k′)-th entry of the transmission matrix Wl is derived from Rayleigh-Sommerfeld diffraction theory.

wk,kl ′ =

dxdy cos χlk,k′ dlk,k′

1 2πdlk,k′ − j

1 λ

ej2πd

l k,k′/λ

where dlk,k′ is the distance between meta-atoms, and χlk,k′ is the angle between the propagation direction and the normal to the layer.

- 3) Air-Ground Channel Model. The channel from the SIM to eVTOL m is modeled as a Rician fading channel. The k-th entry is:

κh κh + 1

ρ0 (dm[n])αh

h¯m[n]

hm,k[n] =

where dm[n] = ∥qm[n] − B∥ is the distance from the ATCo station to the eVTOL, and h¯m[n] = 1 is assumed for the LoS component.

- 4) Received Signal. The composite signal received by eVTOL m at time slot n is:

ym[n] = hHm[n]G[n]

M

m′=1

wm1 ′ pm′[n]sm′[n] + τ

where τ ∼ CN(0, σm2 ) is the complex AWGN.

- 5) Signal-to-Interference-plus-Noise Ratio (SINR). The SINR for eVTOL m at time n is:

SINRm[n] = |hHm[n]G[n]wm1 |2pm[n] M m′=1,m′̸=m |hHm′[n]G[n]wm1 ′|2pm′[n] + σm2

- 6) Achievable Data Rate. The achievable data rate for eVTOL m at time n is given by the Shannon capacity formula:

Rm[n] = log (1 + SINRm[n])

- 7) Joint Optimization Problem (P1). The overall problem is formulated to maximize the sum rate over all eVTOLs and time slots by jointly optimizing power allocation P, phase shifts Ψ, and trajectories Q.

(P1) : max

P,Ψ,Q

g(P, Ψ, Q) =

N

n=1

M

m=1

Rm[n]

s.t.

C1 :

M

m=1

pm[n] ≤ PATC, ∀n ∈ N

C2 : pm[n] ≥ 0, ∀n ∈ N, ∀m ∈ M C3 : θkl [n] ∈ [0, 2π), ∀n, k, l C4 : ∥qm[n] − qm[n − 1]∥ ≤ Vmaxδ, ∀n, m C5 : qm[n] ∈ Rcor, ∀n, m C6 : qm[0] = fm[0], qm[N] = fm[N], ∀m

- 8) Composite Potential Field (CPF) Force. The flight control acceleration for eVTOL i is derived from the negative gradient of the combined potential fields.

ai[n] = −∇ Ficom[n] + Fisep[n] + Fitar[n] The individual fields (target Ftar, separation Fsep, communication Fcom) are functions of the eVTOL’s state and hyperparameters {ktar, ksep, kcom} which are optimized via a DQN framework.

##### E.2 Example 2: Multi-UAV Patrol Inspection with Mobile Edge Computing

This system model captures the complexity of joint communication, computation, and trajectory optimization in UAV-enabled MEC networks.

UAV-MEC System Model

Background This paper considers a multi-UAV patrol inspection system where UAVs traverse predetermined cruise points to collect data and offload it to Ground Base Stations (GBSs) equipped with Mobile Edge Computing (MEC) servers for processing. The system model jointly optimizes cruise point assignment, communication scheduling, computational allocation, and UAV trajectory to minimize total energy consumption and balance task completion times among UAVs.

Key Assumptions • UAVs fly at a constant altitude HU.

- • GBSs are deployed with sufficient density to ensure continuous cellular coverage.
- • A TDMA scheme is used for UAV-GBS communication.
- • The communication rate model incorporates a logistic function of the elevation angle, based on empirical measurements.
- • The information causality constraint must be satisfied (processed data ≤ received data).
- • UAV dynamics follow a rotary-wing energy consumption model.
- • The CPU cycles required per bit (CU) are known and depend on the task type. Parameter Definitions

U = {u1, ..., uN} Set of N UAVs G = {g1, ..., gM} Set of M GBSs S = {s1, ..., sK} Set of K cruise points

wsk ∈ R2×1 Coordinates of cruise point sk (m) wgm ∈ R2×1 Coordinates of GBS gm (m)

HU, HG Altitude of UAV and GBS, respectively (m)

η(t) ∈ R2×1 UAV’s horizontal position at time t (m) v(t) UAV’s velocity vector at time t (m/s); ∥v(t)∥ ≤ Vmax Qsk Data volume collected at cruise point sk (bits)

Rgm(t) Real-time communication rate to GBS gm (bps) τgm(t) ∈ {0, 1} Binary scheduling indicator for GBS gm

fU(t), fgm(t) CPU frequency of UAV and GBS gm, respectively (cycles/s)

CU CPU cycles required per bit (cycles/bit) P(t) UAV transmission power (W)

Ti Task completion time for i-th UAV (s) ϑU UAV’s effective capacitance coefficient (F)

Core Equations 1) Distance and Elevation Angle. The distance between the UAV and a GBS gm at time t is:

dgm(t) = (HU − HG)2 + ∥η(t) − wgm∥2 The corresponding elevation angle is:

HU − HG ∥η(t) − wgm∥ Assumptions: LOS propagation is dominant. UAV and GBS altitudes are constant. Domain: θgm(t) ∈ (0◦, 90◦], dgm(t) > 0.

180 π

θgm(t) ≜

arctan

- 2) Communication Rate Model. The real-time communication rate is given by:

γPˆ (t) (dgm(t))α

χ4 1 + e−(χ1+χ2θgm(t)) H log2 1 +

Rgm(t) = χ3 +

Variables/Constants: χ1, χ2, χ3, χ4 are environment-dependent parameters (χ1 < 0, χ2 > 0, χ4 > 0, χ3 + χ4 = 1). H is the bandwidth (Hz). γˆ = β0/(σ2Λ) is the normalized SNR, where β0 is the reference channel gain (dB), σ2 is the noise power (W), and Λ is the SNR gap. α is the path-loss exponent.

Assumptions: The model accounts for the practical dependence of antenna gain on the elevation angle. Domain: Rgm(t) ≥ 0.

- 3) Information Causal Constraint. The data processed by a GBS cannot exceed the data received from the UAV:

TP 0

fgm(t) CU

dt ≤

TP 0

τgm(t)Rgm(t)dt, ∀TP ∈ [0, Ti] Assumptions: No data buffering at the GBS beyond what is received. Domain: TP ≥ 0.

- 4) Energy Consumption Models. The total energy for the i-th UAV, Ei, is the sum of computation energy (Ec), transmission energy (Et), and flight energy (Ef).

Et =

M

m=1

Ki

k=1

Tsk 0

τgm(t)P(t)dt

Ec =

Ti 0

ϑUfU3 (t)dt

Ef =

Ti 0

 P0 1 +

3∥v(t)∥2 Utip2

+ Pi 1 + ∥v(t)∥4 4v04 −

∥v(t)∥2 2v02

1/2

+

- 1

- 2

d0ρsaˆ∥v(t)∥3

  dt

Ei = Ec + Et + Ef

Variables/Constants: P0, Pi are blade profile and induced power in hover (W). Utip is rotor tip speed (m/s). v0 is mean rotor induced velocity in hover (m/s). d0 is fuselage drag ratio. ρ is air density (kg/m³). s is rotor solidity. aˆ is rotor disc area (m²).

Assumptions: Rotary-wing UAV dynamics. DVFS is used for computation. Domain: Et, Ec, Ef, Ei ≥ 0.

- 5) Original Optimization Problem (P0). The joint optimization problem is formulated as:

N

(Ei + ϕTi + λ(Ti − Tavg))

(P0) : min

{π(k)},{η(t)}, {τgm(t)},tsπ(k),Ti,Ki

i=1

M

τgm(t) ≤ 1 ∀t (9a) Information causal constraint (4) Data processing demand: UAV + GBSs must process all collected data Qsπ(k) Trajectory constraints: Start at sI, visit all points in π, end at sF Velocity constraint: ∥v(t)∥ ≤ Vmax

s.t. τgm(t) ∈ {0, 1},

m=1

- • Variables/Constants: ϕ, λ are compensation factors to balance the dimensions of energy and time in the objective.
- • Assumptions: The problem is decomposed into two tractable subproblems: Task Assignment and Path Planning.
- • Domain: The problem is non-convex and requires decomposition for solution.

##### E.3 Example 3: RIS-Aided Unsourced Random Access RIS-Aided URA System

Background The paper proposes a RIS-aided unsourced random access (URA) system where a massive number of users communicate with a base station (BS) via a reconfigurable intelligent surface (RIS). The direct user-BS links are assumed completely blocked, making the RIS essential for connectivity. The system employs a slotted transmission structure with joint pilot detection, channel estimation, and RIS phase shift optimization to enable reliable communication.

Key Assumptions

- • Quasi-static block fading channels (constant over a frame)
- • Perfect knowledge of RIS-BS channel G (stationary elements)
- • Passive RIS with unit-modulus phase shifts: |[wt]i| = 1
- • Blocked direct user-BS links (no direct path)
- • Saleh-Valenzuela channel model for RIS-BS and user-RIS links
- • UPA antenna arrays at both BS and RIS

Parameter Definitions G ∈ CM×N (RIS-BS channel matrix; Type: Geometric; Unit: dimensionless) hi ∈ C1×N (User-RIS channel vector for user i; Type: Geometric; Unit: dimensionless) wt ∈ CN×1 (RIS phase shift vector at time t; Type: Control; Domain: |[wt]i| = 1) xi,t ∈ C (Transmitted symbol from user i at time t; Type: Information; Unit: dimensionless) zt ∈ CM×1 (Noise vector; Type: AWGN; Distribution: CN(0, σz2IM))

- M (Number of BS antennas; Type: Integer; Unit: dimensionless)
- N (Number of RIS elements; Type: Integer; Unit: dimensionless)

Ka (Number of active users; Type: Integer; Unit: dimensionless) n (Total channel uses; Type: Integer; Unit: dimensionless) LG (Number of paths in RIS-BS channel; Type: Integer; Unit: dimensionless) LR,i (Number of paths in user-RIS channel; Type: Integer; Unit: dimensionless)

Core Equations

###### 1) Received Signal Model (Eq. 4):

Ka

yt =

Gdiag(hi)wtxi,t + zt, t = 1, . . . , n

i=1

Variables: yt ∈ CM×1 (received signal),

G ∈ CM×N, hi ∈ C1×N, wt ∈ CN×1, xi,t ∈ C, zt ∈ CM×1 Assumptions: Blocked direct links, passive RIS, quasi-static channels Domain: |[wt]i| = 1, t ∈ {1, . . . , n}

###### 2) Pilot Phase Received Signal (Eq. 5): Yp = Pp

Gdiag(hi)Wpsdiag(pi) + Zp

i∈Ss

Variables: Yp ∈ CM×np, Wps ∈ CN×np, pi ∈ C1×np, Zp ∈ CM×np Assumptions: Fixed RIS configuration during pilot phase

Domain: |[Wps]i,j| = 1

###### 3) Data Phase Received Signal (Eq. 6):

Yc,f = Pc

Gdiag(hi)Wcsdiag(bi)vi,f + Zc,f

i∈Ss

Variables: Yc,f ∈ CM×ns, Wcs ∈ CN×ns, bi ∈ C1×ns, vi,f ∈ {±1} Assumptions: Two RIS configurations C0 (constant) and C1 (varying) Domain: |[Wcs]i,j| = 1, vi,f ∈ {±1}

###### 4) Channel Model - RIS-BS (Eq. 1):

LG

√

µlaM(ϕr,l, ψr,l)T aN(ϕt,l, ψt,l)

G =

MN

l=1

Variables: µl ∼ CN(0, L0d−l αP L), aM(·), aN(·) (steering vectors) Assumptions: Saleh-Valenzuela model, UPA arrays

Domain: ϕr,l, ψr,l ∈ [0, 2π), ϕt,l, ψt,l ∈ [0, 2π)

###### 5) Channel Model - User-RIS (Eq. 3):

LR,i

√

µfiaN(ϕi,fi, ψi,fi)

hi =

N

fi=1

Variables: µfi ∼ CN(0, L0d−f αP L

) Assumptions: Same path loss model as RIS-BS channel Domain: ϕi,fi, ψi,fi ∈ [0, 2π)

i

###### 6) Steering Vector Model (Eq. 2):

1 √

e−j2πϕ¯n1 ⊗ e−j2πψ¯n2

aN(ϕ, ψ) =

N

Variables: ϕ¯ = sin(ϕ) cos(ψ), ψ¯ = sin(ψ)

d λ

d λ

[0, . . . , N2 − 1] Assumptions: UPA structure with antenna spacing d = λ/2 Domain: ϕ, ψ ∈ [0, 2π)

[0, . . . , N1 − 1], n2 =

n1 =

Model Validation

- • Dimensional consistency: All matrix multiplications are dimensionally consistent (e.g., G ∈ CM×N multiplied by diag(hi) ∈ CN×N yields CM×N matrix)
- • Boundary conditions: Unit-modulus constraint |[wt]i| = 1 enforced for passive RIS
- • Parameter homogeneity: All channel gains µl, µfi have consistent units (dimensionless with path loss scaling)
- • Physical constraints: Angle parameters restricted to [0, 2π), array steering vectors properly normalized

- E.4 Model Extraction Quality Assessment These extracted models demonstrate several quality indicators that validate our automated extraction pipeline:

Completeness: Each model includes comprehensive variable definitions with proper units and domains, ensuring self-contained mathematical descriptions suitable for question generation.

Mathematical Rigor: The extractions preserve complex mathematical relationships including multi-layer matrix products, integral constraints, and summation indices, maintaining the precision required for technical education.

Domain Coverage: The three examples span classical communication theory (Shannon capacity), modern optimization frameworks (joint resource allocation), and emerging technologies (RIS, SIM), reflecting the breadth of WirelessMathBench-XL.

Hierarchical Structure: Models successfully capture equation dependencies, from basic distance calculations to complex optimization objectives, enabling progressive question difficulty design.

#### F Human Expert Evaluation Examples

This section presents representative examples from our expert evaluation process, demonstrating the application of our quality rubric across different score levels. Each example includes the complete question as presented to evaluators, with expert annotations highlighting strengths and weaknesses.

##### F.1 Score 5 - Excellent Quality

Questions scoring 5 demonstrate comprehensive variable definitions, clear mathematical structure, and strong pedagogical value. These questions are ready for immediate use in educational or evaluation contexts.

###### Question ID: 14024 Paper: 2508.03740v1

|LV Q = ∥sg[F] − C∥22 + α ∥F − sg[C]∥22 + βDKL (pc||pu)|
|---|

Answer:

##### Background

In the vector quantization training process, a composite loss function ensures proper codebook learning and feature quantization. The loss consists of three components: codebook loss, commitment loss, and usage regularization, where F ∈ RM×K is the semantic feature matrix, C ∈ RN×K is the codebook matrix, sg[·] denotes the stop-gradient operator, DKL(·) is the Kullback-Leibler divergence, pc is the codeword usage distribution, pu is the uniform distribution, and α,β ∈ R+ are hyperparameters that weight the different loss components.

##### Question

Write the complete vector quantization loss function with all three components.

##### Equation

###### [MASK]

— Please solve this problem step by step. Fill in the [MASK] placeholder(s) with the correct mathematical expression(s). Your final answer should be given at the end in the format:

|your answer<br><br>|
|---|

###### Question ID: 14134 Paper: 2206.08306v1

|mdvdt + 12ρairAfCDv2 + mgr0 cos(α) + mg sin(α) v/ηt + Paccessories ηe<br><br>|
|---|

Answer:

##### Background

The instantaneous fuel rate model calculates the mass flow rate of fuel consumed. Here, m˙ f is the fuel rate (in kg/s), m is the vehicle mass (in kg), dvdt is the acceleration (in m/s²), ρair is the air density (in kg/m³), Af is the frontal area (in m²), CD is the drag coefficient (dimensionless), v is the speed (in m/s), g is gravitational acceleration (in m/s²), r0 is the rolling resistance coefficient (dimensionless), α is the road grade (in radians), ηt is the transmission efficiency (dimensionless), Paccessories is the power consumed by vehicle accessories (in W), and ηe is the engine efficiency (dimensionless).

##### Question

Write the complete equation for the instantaneous fuel rate.

##### Equation

m˙ f = [MASK]

— Please solve this problem step by step. Fill in the [MASK] placeholder(s) with the correct mathematical expression(s). Your final answer should be given at the end in the format:

|your answer<br><br>|
|---|

###### Question ID: 4149

|√PxWxx + √PzWzz + Wnn<br><br>|
|---|

Paper: 2505.19983v1 Answer:

##### Background

In a wireless semantic communication system with interference, the received real-valued signal after equalization combines the desired signal, an interference signal, and noise. The system model is derived from the complex baseband representation, where y ∈ R2k is the equalized received real signal vector, x ∈ R2k is the real-valued semantic feature vector to be transmitted, z ∈ R2k is the real-valued interference vector, n ∼ N(0, σ

2

2 I2k) is the real-valued additive white Gaussian noise vector, Px ∈ R+ is the desired signal transmit power (in linear scale), Pz ∈ R+ is the interference signal transmit power (in linear scale), Wx ∈ R2k×2k is the channel transformation matrix for the desired signal, Wz ∈ R2k×2k is the channel transformation matrix for the interference signal, and Wn ∈ R2k×2k is the channel transformation matrix for the noise.

##### Question

Write the complete received signal equation including all three components.

##### Equation

###### y = [MASK]

— Please solve this problem step by step. Fill in the [MASK] placeholder(s) with the correct mathematical expression(s). Your final answer should be given at the end in the format:

|your answer<br><br>|
|---|

##### F.2 Score 4 - Good Quality

Questions scoring 4 contain solid technical content with minor areas for improvement, typically in completeness of context or clarity of problem statement.

###### Question ID: 14101

|arctan<br><br>hi r<br><br>|
|---|

Paper: 2208.11967v1 Answer:

##### Background

In a laser-powered UAV-assisted wireless network, the probability of having a Line-of-Sight (LOS) link is crucial for signal propagation. This model characterizes the LOS probability between an aerial or terrestrial node and a user, where Pi(r) is the probability of an LOS link for node type i (where i ∈ {Lu,Lb} representing LOS UAV and LOS TBS links, respectively), r is the horizontal distance between the transmitter and receiver (in meters), hi is the altitude or height of the node type i (in meters), and a, b, c are environment-dependent parameters (dimensionless) that model the blockage characteristics in urban, suburban, or dense urban environments.

##### Question

What trigonometric function of the elevation angle is the argument of the exponential?

##### Equation

Pi(r) = −aexp(−b[MASK]) + c

— Please solve this problem step by step. Fill in the [MASK] placeholder(s) with the correct mathematical expression(s). Your final answer should be given at the end in the format:

|your answer<br><br>|
|---|

###### Question ID: 4439

|B(1 + νk,i) µ ln 2 −<br><br>N0,k + Ik,i λk,i<br><br>+|
|---|

Paper: 2506.01400v1

Answer:

##### Background

The optimal power allocation for communication UEs in a multi-user MIMO system is derived using the Karush-Kuhn-Tucker (KKT) conditions to solve the constrained optimization problem. This solution follows a water-filling structure. Here, PC,k,i is the optimal power allocated to the i-th sub-channel of communication UE k (in W), [·]+ = max(0,·) ensures non-negative power, B is the bandwidth (in Hz), νk,i is the Lagrange multiplier associated with the minimum capacity constraint for the i-th sub-channel of UE k (dimensionless), µ is the Lagrange multiplier associated with the total power constraint (in W−1), N0,k is the noise power at UE k (in W), Ik,i is the interference power (in W), and λk,i is the channel gain eigenvalue (dimensionless).

##### Question

Write the complete optimal power allocation formula for a communication user equipment (UE).

##### Equation

PC,k,i = [MASK]

— Please solve this problem step by step. Fill in the [MASK] placeholder(s) with the correct mathematical expression(s). Your final answer should be given at the end in the format:

|your answer<br><br>|
|---|

##### F.3 Score 3 - Acceptable Quality

Questions scoring 3 meet minimum requirements but have noticeable gaps in clarity or completeness that limit their educational value.

Question ID: 13890 Paper: 2208.07045v1 Answer: A

##### Background

In an interference-coupled multi-cell RAN slicing system, the Signal-to-Interference-plus-Noise Ratio (SINR) is calculated at a specific user location. The SINR determines the quality of the wireless link for a user served by a particular channel in a slice, where γs,q(l,∆s,q) is the SINR at location l for channel q in slice s (dimensionless), Ps,qSI (l) is the received signal power at location l from the base station transmitting on channel q in slice s (in watts), Ns,q is the set of all slice-channel pairs that can potentially interfere with (s,q) (dimensionless), ∆s,q is a binary vector indicating which interfering transmitters in Ns,q are active (dimensionless), P(INs′,q′),(s,q)(l) is the interference power at location l from an interfering transmitter on channel q′ in slice s′ (in watts), and N0 is the noise power (in watts).

##### Question

Which expression correctly represents the SINR calculation that should replace [MASK]?

##### Equation

γs,q(l,∆s,q) = [MASK]

##### Options

Ps,qSI (l)

###### • A:

∆s,q(s′, q′)P(INs′,q′),(s,q)(l) + N0

(s′,q′)∈Ns,q\(s,q)

Ps,qSI (l)

###### • B:

∆s,q(s′,q′)P(INs′,q′),(s,q)(l) + N0

(s′,q′)∈Ns,q

Ps,qSI (l)

###### • C:

P(INs′,q′),(s,q)(l) + N0

(s′,q′)∈Ns,q\(s,q)

∆s,q(s′,q′)P(INs′,q′),(s,q)(l)

′,q′)∈Ns,q\(s,q)

###### • D: (s

Ps,qSI (l) + N0

|X|
|---|

###### • Your final answer should be given at the end in the format:

Question ID: 4275 Paper: 2504.18155v1 Answer: A

##### Background

In the hierarchical cell-free massive MIMO uplink training phase, edge access points (eAPs) receive pilot sequences from multiple users. The received pilot signal matrix at eAP l combines contributions from all users through their respective channels, where Ψl ∈ CN

a×τp represents the received pilot signal matrix at eAP l, pu is the user transmit power constraint (in watts), K = {1,...,K} is the set of user indices, hkl ∈ CN

p×1 is the pilot sequence of user k (dimensionless), Zl ∈ CN

a×1 is the channel vector from user k to eAP l, ik ∈ Cτ

a×τp is the additive noise matrix with entries ∼ CN(0,σz2), Na is the number of antennas per eAP, and τp is the pilot sequence length (in symbols).

##### Question

Which expression correctly represents the received pilot signal matrix at eAP l?

##### Equation

Ψl = [MASK]

##### Options

- • A: √pu k∈K hkliTk + Zl

- • B: √pu k∈K hkliHk + Zl

- • C: √pu k∈K hTklik + Zl

- • D: √pu k∈K hHklik + Zl

- • Your final answer should be given at the end in the format:

|X|
|---|

##### F.4 Score 2 - Poor Quality

Questions scoring 2 have significant deficiencies that impair their usefulness, though they may contain salvageable elements.

###### Question ID: 13936

|ϕ|
|---|

Paper: 2502.11053v2 Answer:

##### Background

In the belief propagation decoding of LDPC codes, messages are passed between nodes on the Tanner graph. For the check node update, L(rji) ∈ R is the log-likelihood ratio (LLR) message sent from check node j to bit node i. L(qi′j) ∈ R is the LLR message received from a connected bit node i′. The set BNj\i contains all bit nodes connected to check node j except bit node i.

##### Question

Which function is applied to the absolute value of each incoming LLR before summation in the stable SPA update?

##### Equation

 

  · ϕ

 

 

L(rji) =

sign(L(qi′j))

[MASK]

i′∈BNj\i

i′∈BNj\i

— Please solve this problem step by step. Fill in the [MASK] placeholder(s) with the correct mathematical expression(s). Your final answer should be given at the end in the format:

|your answer<br><br>|
|---|

###### Question ID: 4173

|I(t)|
|---|

Paper: 2505.18534v1 Answer:

|sin|
|---|

,

,

|cos|
|---|

##### Background

In a DSP-free coherent optical interconnect system using offset-QAM modulation, the received in-phase and quadrature signals are processed before carrier phase recovery. The system aims to compensate for a phase error between the received signal and the local oscillator. Here, I′(t) represents the received in-phase signal after mixing and before phase correction (in volts or amperes), Q′(t) is the corresponding quadrature signal (in volts or amperes), I(t) ∈ {±AOMA/2} is the original modulated in-phase data signal (in volts or amperes), Q(t) ∈ {±AOMA/2} is the original modulated quadrature data signal (in volts or amperes), A0 ∈ R+ is the constant DC offset introduced by the offset-QAM modulation format (in volts or amperes), and ∆ϕ ∈ (−π,π] is the phase error between the transmitter and local oscillator paths (in radians).

##### Question

Complete the three missing components: the data signal and the two trigonometric functions.

##### Equation

I′(t) = ([MASK] + A0)[MASK](∆ϕ) + (Q(t) + A0)[MASK](∆ϕ)

— Please solve this problem step by step. Fill in the [MASK] placeholder(s) with the correct mathematical expression(s). Your final answers should be given at the end in the format:

|answer1|
|---|

|answer2|
|---|

,

, ... (for the 3 blanks in order)

##### F.5 Score 1 - Very Poor Quality

Questions scoring 1 have fundamental errors or omissions that render them unusable without complete revision.

###### Question ID: 13863

|log|
|---|

Paper: 2412.01187v1 Answer:

##### Background

In a point-to-point, interference-free multi-terminal wireless system with NU single-antenna users communicating over parallel links, the instantaneous achievable rate is modeled for each link. The rate for terminal i is a function of the channel state and the allocated power, where ri(pi(h),hi) represents the instantaneous achievable rate on link i (in bps/Hz), pi(h) is the power allocated to terminal i for a given channel realization h (in watts), hi is the fading channel coefficient for terminal i (dimensionless), and σi2 is the noise variance on link i (in watts). The system assumes AWGN channels and perfect Channel State Information (CSI).

##### Question

What is the outer function that transforms the SNR into a rate?

##### Equation

pi(h) · h2i σi2

ri(pi(h),hi) ≜ [MASK] 1 +

— Please solve this problem step by step. Fill in the [MASK] placeholder(s) with the correct mathematical expression(s). Your final answer should be given at the end in the format:

|your answer<br><br>|
|---|

###### Question ID: 4264

|K<br><br>k=1<br><br>∥tHMS ⊙ H:,k∥2|
|---|

|∥vec(H)∥2|
|---|

Paper: 2504.21128v1 Answer:

=

##### Background

The HMA system must obey a global power conservation constraint for the passive metasurface. This constraint links the HMS configuration to the communication channel. vec(H) ∈ CNK×1 is the vectorized channel matrix, tHMS ∈ CN×1 is the vector of complex transmission coefficients for the N HMS unit-cells, H:,k ∈ CN×1 is the k-th column of the channel matrix (representing the channel from user k to all unit-cells), and ⊙ denotes the Hadamard (element-wise) product.

##### Question

Write the complete global power conservation constraint equation.

##### Equation

###### [MASK] = [MASK]

— Please solve this problem step by step. Fill in the [MASK] placeholder(s) with the correct mathematical expression(s). Your final answers should be given at the end in the format:

|answer1|
|---|

|answer2|
|---|

,

, ... (for the 2 blanks in order)

#### G Representative Solution Examples from WirelessMathLM-7B

This appendix presents detailed examples of solutions generated by WirelessMathLM-7B, organized by question type and quality level. These examples illustrate the model’s reasoning patterns, mathematical sophistication, and common error modes identified in our qualitative analysis.

##### G.1 High-Quality Solution Examples

###### G.1.1 Multiple Choice Question: Matrix All-Pass Filter Question ID: 11325 Correct Answer: B Model Response: B

WirelessMathLM-7B Solution

To determine the correct factorization of the matrix all-pass filter G(z), we need to understand the properties and definition of a matrix all-pass filter. A matrix all-pass filter is a filter whose frequency response has a magnitude of 1 for all frequencies, but its phase response can vary. Mathematically, a matrix all-pass filter can be represented as:

G(z) = N(z)D−1(z)

where N(z) is the numerator matrix polynomial and D(z) is the denominator matrix polynomial. The matrix all-pass filter is defined such that:

G(z)G−1(z) = Im

Given that D0 = Im, the filter is causal and proper. This means that the denominator polynomial D(z) is a proper polynomial (its degree is less than or equal to m − 1). To verify that the given factorization is correct, we can check the inverse of G(z):

G−1(z) = (N(z)D−1(z))−1 = D(z)N−1(z)

This is because the inverse of a product of matrices is the product of their inverses in reverse order. Now, we can check the product G(z)G−1(z):

G(z)G−1(z) = (N(z)D−1(z))(D(z)N−1(z)) = N(z)D−1(z)D(z)N−1(z) = N(z)N−1(z) = Im

This confirms that G(z) is indeed an all-pass filter. Therefore, the correct factorization of the matrix all-pass filter G(z) is:

G(z) = N(z)D−1(z) Thus, the correct answer is:

|B|
|---|

Analysis: This solution demonstrates sophisticated understanding of matrix theory and filter design. The model correctly identifies the all-pass property, provides mathematical verification through matrix inverse operations, and connects the factorization to causality constraints. The reasoning is systematic, mathematically rigorous, and includes proper justification for each step.

###### G.1.2 Fill-in-the-Blank (100%): Cell-Free Massive MIMO Beamforming

|Pm<br><br>K<br><br>k=1<br><br>√ηmkgˆmk∗ uk<br><br>|
|---|

###### Question ID: 18369 Correct Answer:

WirelessMathLM-7B Solution

To derive the conjugate beamforming transmitted signal in a Cell-Free Massive MIMO (CFmMIMO) system, we need to consider the linear combination of the users’ data symbols, precoded using the locally estimated channel state information. The signal transmitted from the m-th AP is given by:

ηmkgˆmk∗ uk

sm =

k∈K

where: - sm ∈ C is the transmitted symbol from the m-th AP (dimensionless), - Pm ∈ R+ is the maximum transmit power of the m-th AP (in watts, W), - ηmk ∈ R+ is the power control coefficient for user k at AP m (dimensionless), - gˆmk ∈ C is the estimated channel coefficient from AP m to user k (dimensionless), - uk ∈ C is the information symbol intended for user k (dimensionless; E[|uk|2] = 1). The term gˆmk∗ represents the complex conjugate of the estimated channel coefficient from AP m to user k. This is because conjugate beamforming is used to cancel out the phase shifts introduced by the channel. Therefore, the complete equation for the conjugate beamforming transmitted signal is:

ηmkgˆmk∗ uk

sm =

k∈K

|k∈K<br><br>ηmkgˆmk∗ uk|
|---|

So, the final answer is:

Analysis: This solution exhibits deep understanding of MIMO systems, correctly identifying conjugate beamforming principles and providing clear physical intuition. The model properly handles complex notation, explains the rationale for complex conjugation, and maintains dimensional consistency throughout the derivation.

###### G.1.3 Fill-in-the-Blank (50%): Gaussian Function Components

|(λ − λp)2|
|---|

|∆λ2|
|---|

Question ID: 5582 Correct Answer:

and WirelessMathLM-7B Solution (Truncated)

To solve the problem, we need to understand the form of the Gaussian function and how it is modified to include the secondary skewed Gaussian component. The given Gaussian function is:

[MASK] [MASK]

g(λ,λp,∆λ) = exp −

The standard form of a Gaussian function centered at λp with a width parameter ∆λ is:

(λ − λp)2 2(∆λ)2 However, the problem involves a secondary skewed Gaussian component, which is typically represented by a function that is asymmetric around the peak wavelength λp... Given the form of the Gaussian function, the [MASK] placeholder in the exponent should be filled with (λ − λp)2. Therefore, the final answer is:

g(λ,λp,∆λ) = exp −

|(λ − λp)2|
|---|

|(∆λ)2|
|---|

,

Analysis: The model correctly identifies standard Gaussian form and provides appropriate mathematical expressions. The reasoning demonstrates understanding of probability density functions and their parameters in optical communication contexts.

##### G.2 Error Analysis Examples

###### G.2.1 Mathematical Equivalence Error Question ID: 2406 Type: Mathematical Equivalence Failure

|G 2 − 1<br><br>|
|---|

|G|
|---|

Correct Answer:

Model Response:

Error Example: Mathematical Equivalence

To determine the exponent of the sinusoidal term that defines the pattern’s shape, we need to analyze the given equation for the normalized power radiation pattern of a single IRS element:

(sinθ cosφ)[MASK], θ ∈ [0,π],φ ∈ [−π2, π2], 0, otherwise.

F(Π) ≜

The equation suggests that the power radiation pattern is a function of the angles θ (elevation angle) and φ (azimuth angle). The term sinθ cosφ is a product of two sinusoidal functions, and the exponent of this product is what we need to identify. Given that the problem involves a sinusoidal term, we can infer that the exponent is likely a constant that determines the shape of the pattern. Since the problem does not provide any additional information about the specific shape or the value of the exponent, we can assume that the simplest form of the pattern is a first-order sinusoidal function. Therefore, the exponent of the sinusoidal term is G, where G is the peak power gain of a single IRS element. This means that the power radiation pattern is proportional to the square of the sine of the elevation angle and the cosine of the azimuth angle, raised to the power of G. Thus, the exponent of the sinusoidal term is

|G|
|---|

.

Error Analysis: The model provides reasonable physical interpretation but fails to derive the precise mathematical relationship G2 − 1. This represents a common error type where domain knowledge is correctly applied but mathematical transformation is incomplete. The model recognizes that G relates to the exponent but doesn’t perform the necessary algebraic manipulation.

###### G.2.2 Conceptual Misunderstanding Error Question ID: 16144 Type: Conceptual Misunderstanding

|M<br><br>m=1<br><br>Ki<br><br>k=1<br><br>Tsk<br><br>0<br><br>τg<br><br>m<br><br>(t)P(t)dt|
|---|

###### Correct Answer:

Error Example: Conceptual Misunderstanding

To determine the total transmission energy Et for a UAV, we need to consider the energy consumed by each transmission to each GBS. The energy consumed by a transmission is given by the product of the transmission power and the time spent transmitting. Therefore, the total transmission energy can be calculated by summing up the energy consumed for all transmissions to all GBSs.

The energy consumed by a transmission from the i-th UAV to the m-th GBS is P(t) · Ts

(t), where P(t) is the transmission power, Ts

k · τg

m

is the time spent at the k-th cruise point, and τg

(t) is the binary

m

k

scheduling variable... Thus, the total transmission energy Et is given by:

Ki

N

M

P(t) · Ts

k · τg

Et =

(t)

m

m=1

i=1

k=1

|N<br><br>i=1<br><br>Ki<br><br>k=1<br><br>M<br><br>m=1<br><br>P(t) · Ts<br><br>k · τg<br><br>m<br><br>(t)|
|---|

The final answer is:

Error Analysis: This error demonstrates correct energy calculation principles but incorrect mathematical formulation. The model understands that energy equals power times time but fails to recognize the need for temporal integration rather than discrete summation. The error reflects misunderstanding of continuous vs. discrete system modeling rather than fundamental energy concepts.

###### G.2.3 MCQ Selection Error Question ID: 16315 Type: Multiple Choice Selection Correct Answer: B Model Response: C

Error Example: MCQ Selection (Truncated)

To determine the correct term that completes the equation for the phase compensation factor βℓ, we need to understand the physical significance of the phase compensation term. The phase compensation term βℓ accounts for the path difference that is corrected for in the virtual array projection... The total phase difference ∆ϕ is the sum of these two phase differences:

2π (dxℓ)2 + (dyℓ)2 λ

2πdzℓ λ

∆ϕ = ∆ϕz + ∆ϕhorizontal =

+

To correct for this phase difference, we need to add the phase compensation term βℓ:

(dxℓ)2 + (dyℓ)2 2Z0

βℓ = 2dzℓ +

|C|
|---|

Therefore, the correct term that completes the equation is:

Error Analysis: This example shows mathematically sound reasoning leading to an incorrect final selection. The model provides correct physical interpretation and mathematical derivations but selects the wrong multiplechoice option. This suggests challenges in mapping derived expressions to provided answer choices rather than fundamental understanding failures.

