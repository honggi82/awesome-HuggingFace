## arXiv:2602.09443v1[cs.AI]10Feb2026

P1-VL Technical Report 2026-02-11

# P1-VL: Bridging Visual Perception and Scientific Reasoning in Physics Olympiads

Yun Luo∗ †, Futing Wang∗, Qianjia Cheng∗, Fangchen Yu∗, Haodi Lei, Jianhao Yan, Chenxi Li, Jiacheng Chen, Yufeng Zhao, Haiyuan Wan, Yuchen Zhang, Shenghe Zheng, Junchi Yao, Qingyang Zhang, Haonan He, Wenxuan Zeng, Li Sheng, Chengxing Xie, Yuxin Zuo, Yizhuo Li, Yulun Wu, Rui Huang, Dongzhan Zhou, Kai Chen, Yu Qiao, Lei Bai , Yu Cheng , Ning Ding , Bowen Zhou , Peng Ye , Ganqu Cui †

P1 Team, Shanghai AI Laboratory

∗ Equal Contribution Corresponding Authors † Technical Leads

cuiganqu@pjlab.org.cn, luoyun1@pjlab.org.cn P1-VL Tech Blog

Abstract | The transition from symbolic manipulation to science-grade reasoning represents a pivotal frontier for Large Language Models (LLMs), with physics serving as the critical test anchor for binding abstract logic to physical reality. Physics demands that a model maintain physical consistency with the laws governing the universe, a task that fundamentally requires multimodal perception to ground abstract logic in reality. At the Olympiad level, diagrams are often constitutive rather than illustrative, containing essential constraints, such as boundary conditions and spatial symmetries, that are absent from the text. To bridge this visual-logical gap, we introduce P1-VL, a family of open-source vision-language models engineered for advanced scientific reasoning. Our method harmonizes Curriculum Reinforcement Learning, which employs progressive difficulty expansion to stabilize post-training, with Agentic Augmentation, enabling iterative self-verification at inference. Evaluated on HiPhO, a rigorous benchmark of 13 exams from 2024–2025, our flagship P1-VL-235B-A22B becomes the first open-source Vision-Language Model (VLM) to secure 12 gold medals and achieves the state-of-the-art performance in the open-source models. Our agent-augmented system achieves the No.2 overall rank globally, trailing only Gemini-3-Pro. Beyond physics, P1-VL demonstrates remarkable scientific reasoning capacity and generalizability, establishing significant leads over base models in STEM benchmarks. By open-sourcing P1-VL, we provide a foundational step toward general-purpose physical intelligence to better align visual perceptions with abstract physical laws for machine scientific discovery.

###### High School Physics Olympiad Benchmark (HiPhO)

45

42.7

AverageExamScore(Theory)

Open-source

Closed-source Medals 13 8

40.9 40.6

| |
|---|

39.3

40

38.4 37.7 37.4 36.9

| | | | | |
|---|---|---|---|---|

35.9 35.8 35.3 35.0

33.9 33.5

35

32.5 32.4 32.1 31.7

30.0 29.7

30

25

20

P1-VL-235B-A22BGPT-5.2(high)P1-235B-A22B+

P1-VL-235B-A22B+Gemini-3-Pro(high)PhysicsMinions

Gemini-2.5-ProDeepSeek-V3.2-ThinkingGPT-5 P1-235B-A22BGrok-4P1-VL-30B-A3Bo3Qwen3-VL-235B

Qwen3-VL-30B-A3B

-ThinkingDeepSeek-R1Claude-4-Sonnet

P1-30B-A3BGPT-OSS-120B(high)Gemini-2.5-Flash

-A22B-ThinkingQwen3-235B-A22B-Thinking-2507

PhysicsMinions

-Thinking

-Thinking

Figure 1 | P1-VL-235B-A22B stands as the state-of-the-art open-source VLM in the Physics Olympiad benchmark (HiPhO), placing No.3 behind Gemini-3-Pro(high) and GPT-5.2(high) and achieving 12 gold medals. Even at mid-scale, P1-VL-30B-A3B achieved 9 gold medals, with a higher average score than most of the open-source models except P1-235B-A22B and DeepSeek-V3.2-Thinking. With the PhysicsMinions agent framework, P1-VL-235B-A22B+PhysicsMinions ranks No.2 on HiPhO.

### Contents

- 1 Introduction 3
- 2 Physics Dataset 5

- 2.1 Overview . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 5
- 2.2 Dataset Construction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 5
- 2.3 Quality Control. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 6

- 3 Approach 6

- 3.1 RL Formulation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 6
- 3.2 Instantiation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 7
- 3.3 Technical Design . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 8

- 3.3.1 Curriculum Training . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 8
- 3.3.2 Training Stabilization Mechanism . . . . . . . . . . . . . . . . . . . . . . . . . 9

- 3.4 Training Dynamics . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 10
- 3.5 Agentic Augmentation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 11

- 4 Experiment 11

- 4.1 Experimental Setup . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 11
- 4.2 Evaluation on Physics Olympiads . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 13

- 5 Discussion 15

- 5.1 Generalizability of P1-VL . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 15
- 5.2 Train-inference Mismatch . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 17
- 5.3 Effect of Training Data Composition . . . . . . . . . . . . . . . . . . . . . . . . . . . . 17
- 5.4 Effect of Curriculum Training . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 17

- 6 Conclusion 18
- 7 Acknowlegement 18

A Appendix 26

- A.1 RL Training on Intern Series . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 26
- A.2 Case Study I: Text+Variable Figure . . . . . . . . . . . . . . . . . . . . . . . . . . . . 26

### 1. Introduction

Recent advances in Large Language Models (LLMs) (Comanici et al., 2025; Yang et al., 2025) have propelled artificial intelligence beyond symbolic manipulation toward science-grade reasoning (Bai et al., 2025a; Gibney, 2025; Zhang et al., 2024). Among scientific disciplines, physics stands as the critical test anchor for these emerging capabilities. Physics inherently demands a rigorous synthesis of multimodal perception and causal logic to model the behavior of the universe. Consequently, the ability to solve complex physics problems, a task that often requires mapping physical intuition to abstract laws, provides a rigorous test for genuine, first-principles reasoning.

Olympiad-level competitions, such as the International Physics Olympiad (IPhO) serve as a significant surrogate to evaluate the physics reasoning capacity of language models, where problems demand deep conceptual understanding and precise system decomposition. While recent works have begun to adapt LLMs for these challenges (Chen et al., 2025b; Qiu et al., 2025; Yu et al., 2025b), they predominantly focus on textual reasoning, overlooking a fundamental characteristic of physical problem-solving: its inherent multimodality. In many Olympiad problems, diagrams are not merely illustrative but constitutive—they contain essential geometric constraints, circuit topologies, and force interactions that are purposefully omitted from the text. Without the ability to perceive and interpret these visual schematics, a text-only model faces an insurmountable information gap. As illustrated in Figure 2 (IPhO 2025), the task involves quantifying bubble radii and ascent velocities directly from visual inputs, effectively simulating the data analysis process in real-world physics experiments. Thus, Physics Olympiads serve as high-fidelity testbeds for Vision-Language Models (VLMs), verifying the critical ability to align visual observations with abstract physical laws (Yu et al., 2025a).

###### IPhO 2025 Q3 - Part A:

|Champagne is a French sparkling wine. Fermentation of sugars produces carbon dioxide (CO ) in the bottle. The molar concentration of CO in the liquid phase 𝑐ℓ and the partial pressure 𝑃 in the gas phase are related by 𝑐ℓ = 𝑘 𝑃 known as Henry's law and where 𝑘 is called Henry's constant.<br><br>• Surface tension of champagne 𝜎 = 47×10 𝐽 ⋅ 𝑚 ,<br>• Density of the liquid 𝜌ℓ = 1.0×10 𝑘𝑔 ⋅ 𝑚 ,<br>• Atmospheric pressure 𝑃 = 1 𝑏𝑎𝑟 = 1.0×10 𝑃𝑎.<br><br><br>......<br><br>......The bubbles of the train have the same initial radius and are emitted at a constant frequency 𝑓b = 20 Hz. For the range of velocities studied here, the drag force 𝐹 on a bubble of radius 𝑎 moving at velocity 𝑣 in a liquid of dynamic viscosity 𝜂 is given by Stokes‘ law 𝐹 = 6𝜋𝜂𝑎𝑣 ......<br><br>[Figure 1]<br><br>Fig. 1. A glass filled with champagne.|
|---|
|Fig. 2. A train of bubbles. The photo is rotated horizontally for the page.<br><br>Question: Give the expression of the main forces exerted on a vertically rising bubble. Obtain the expression of 𝑣(𝑎). Give a numerical estimate of 𝜂 using 𝜌ℓ, 𝑔 and quantities measured on Fig. 2.<br><br>[Figure 2]|
|Noted:Requires measuring the radius 𝒂 of the bubbles in the photograph (Fig. 2), and measuring the ascent velocity 𝒗 of the bubbles, i.e. the vertical distance z between bubble n and bubble n-2 on the photograph to derive the velocity.|

Figure 2 | A question sample from the International Physics Olympiad 2025 (IPhO 2025), where the question requires measuring the radius of bubbles and estimating their velocity in Fig 2.

Successfully bridging this visual-logical gap is more than an academic achievement; it lays the foundation for general-purpose physical intelligence. We posit that mastering the rigorous constraints of Olympiad problems is a necessary precursor to achieving machine scientific discovery (Zheng et al., 2025c) and reliable embodied AI. Just as a scientist needs a theory and a robot needs a world model,

future AI systems must first internalize the laws of physics in a controlled environment. Therefore, our focus on Olympiad-level reasoning is not an end in itself, but a strategic step toward building systems capable of understanding and eventually navigating the physical reality.

In this work, we introduce P1-VL, a new family of open-source vision-language models designed to advance the frontier of scientific AI. Our framework harmonizes two critical paradigms: Curriculum Reinforcement Learning (RL) Training and Agentic Augmentation. This dual approach ensures that the model not only internalizes robust reasoning capabilities through RL (Sutton et al., 1998), but also effectively operationalizes them via adaptive agentic control during inference (Feng et al., 2026).

- • Curriculum RL Training. P1-VL is developed exclusively through RL post-training (Cui et al.,

- 2025a; Guo et al., 2025) built upon base vision-language models. To maximize the efficacy of this paradigm, we introduce a curriculum RL framework that systematically advances reasoning capabilities through progressive difficulty expansion, and robust stabilization mechanisms. By ensuring a structured learning trajectory, our design enables sustained, long-term optimization and effectively circumvents notorious RL pitfalls such as reward sparsity, entropy collapse, and training stagnation.

- • Agentic Augmentation. During the inference phase, we integrate P1-VL models with the PhysicsMinions agent framework (Yu et al., 2025b) to enable iterative correction and self-verification. This mechanism facilitates multi-turn reflection, empowering the model to reason, critique, and refine its solutions—mirroring the rigorous cognitive process of human physicists. By leveraging this structured test-time reasoning, P1-VL effectively extends its problem-solving depth and reliability without requiring additional training parameters.

We release two variants of the P1-VL family and evaluate them on HiPhO (Yu et al., 2025a), a rigorous benchmark aggregating 13 recent Olympiad exams from 2024–2025. Our flagship model, P1-VL-235B-A22B, marks a significant milestone for the open-source community: it is the first specialized physics vision-language model to secure 12 golds and 1 silver across the full HiPhO suite. Remarkably, the standalone P1-VL surpasses even the agent-enhanced text baseline (P1-235B-A22B+PhysicsMinions (Chen et al., 2025b)), ranking 3rd among all open and closedsource models. Meanwhile, our lightweight variant, P1-VL-30B-A3B, achieves 9 gold medals, surpassing the average score of most open-source models. When further augmented with the PhysicsMinions agent framework, P1-VL ascends to the 2nd overall spot on the leaderboard.

Beyond its specialized domain, P1-VL demonstrates remarkable generalizability. In the multidisciplinary FrontierScience-Olympiad, both P1-VL-235B-A22B and P1-VL-30B-A3B models demonstrate robust gains across biology, chemistry, and physics, outperforming their respective baselines by margins of 8.0 and 9.1 points, respectively. In the mathematical reasoning tasks and multi-modal STEM reasoning tasks, our model could also consistently outperform their base models. This superiority extends to out-of-domain evaluations, where both variants consistently eclipse their base models across diverse text, and multi-modal benchmarks.

Contributions. This work makes the following key contributions:

- • We introduce P1-VL, a first family of open-source Vision-Language Model specialized in physics problems, achieving 12 gold medals in physics Olympiad competitions.
- • We develop a curriculum RL framework with difficulty expansion, and training stabilization for sustained reasoning improvement.
- • We open-source the P1-VL model family, filling the multimodal void within the P1 ecosystem (Chen et al., 2025b), and bridging the gap between visual perception and scientific reasoning.

Collectively, the progress lays the foundation for AI systems that not only interpret reality but eventually contribute to the frontiers of physics research and the development of robust world models.

Statistics Number Data Composition Total Problems 8,033

4 Modality Types

16.5%

31.4% Text+Illustration Fig.

Text+Variable Fig.

- - Problems from Olympiads 4,126 (51%)
- - Problems from textbooks 3,907 (49%)

Text+Data Fig.

36.8%

Total Answers 10,516 Total Fields 5 Total Subfields 25 Total Answer Types 5

15.3%

Text-Only

5 Physics Fields

22.5%

Mechanics

Data Sources

Electromagnetism

14.1%

Textbooks 20 Olympiad Types 9

Thermodynamics

10.0% 8.8%

44.6%

Modern Physics

- Sets Collected 199 Token Statistics

Optics

Average Question Tokens 943 Max Question Tokens 6,604 Max Solution Tokens 5,519

6 Answer Types

23.4% 27.6%

Expression

Numerical Value

Images Statistics

Equation

Questions with Images 5,513 Questions without Images 2,520 Max Images per Question 8 Average Images per Question 0.82

26.4%

Multi-Choice

21.9%

Inequality

- Table 1 | Statistics of the multi-modal training data.

Figure 3 | Distribution of the training data.

### 2. Physics Dataset

###### 2.1. Overview

We introduce a systematically curated multimodal dataset of 8,033 problems designed to advance genuine scientific reasoning in VLMs. The dataset includes problems from physics Olympiads (4,126), undergraduate textbooks (2,968), and competition guides (939), as summarized in Table 1.

###### 2.2. Dataset Construction

Data Collection. Our data construction is driven by a core objective: enabling VLMs to internalize structured reasoning aligned with physical laws and empirical consistency—a prerequisite for genuine scientific intelligence. To achieve this, we prioritize problems that combine conceptual depth with ruleverifiable outcomes, traits shown to catalyze reasoning improvements (Chen et al., 2025b; Guo et al., 2025; Wen et al., 2025; Yang et al., 2025). Within the physics domain, physics Olympiad problems naturally contain such properties: they demand rigorous modeling and multi-step inference while remaining formally tractable. The value of this specific difficulty profile is underscored by empirical findings in PHYSICS (Zheng et al., 2025b), which demonstrate that LLMs face greater challenges in Olympiad tasks than in introductory undergraduate problems. Accordingly, we first curate our dataset from two complementary pillars: ten prestigious physics Olympiads (up to 2023, including IPhO and APhO) that capture a graded difficulty spectrum, and ten authoritative competition textbooks that provide systematically organized exercises with expert-verified solutions. To further augment the dataset’s multi-modal richness, we extend our collection to include undergraduate textbooks. These sources serve as a vital repository of image-centric problems, significantly enhancing the overall density of visual schematics and diagrams within the corpus.

Question

As the old proverb goes, "seeing is believing, and since seeing depends on light, what we see is highly dependent on the nature of light. These following problems aim to investigate the peculiarities of vision in the light of the Relativity theory, particularly those arising from the effect of light's nonzero travel time. Consider a simplified view of image formation in a camera, wherein said device works by capturing light from an object and projecting it onto a screen. If the object is far away, light from it would obviously be focused at the focal point of the cameras lens. It is easily shown that the size of the image formed on the screen of the camera, for an object of size H at a distance L from the camera, is given by h=Hf/L. <Image> Suppose now that the camera is approaching the object at a relativistic speed v while rapidly taking a photograph; assume the distance between the camera and the object is still L. At what rate will the height of the image on the screen be changing at this instant?

- PDF 1

- PDF 2

- Markdown 1

- Markdown 2

PDF Conversion

...

...

PDF n

Markdown n

QA Parsing

Question 1 Answer 1

Question 2 Answer 2

Question n Answer n

...

Answer Annotation

[Figure 3]

Question: As the old proverb goes, "seeing is believing, and since ... Answer:[\frac{Hfv}{L^{2}}\....]

Diagram:

[Figure 4]

Post Processing

[Figure 5]

Language Transformation

...... Expert Review

|Physics Field<br><br>Source<br><br>Optics<br><br>Modality Type<br><br>WoPhO Text+Variable Figure<br><br>Rule-Verifiable Answers<br><br>Answer: \frac{Hfv}{L^{2}}\frac{1}{1+v/c} Answer Type: Expression Answer Unit: ----|
|---|

Final Data

Multi-Modal Training data (Question1, Answer1, Images1, Metadata1) …

- Figure 4 | Data collection pipeline for physics data.

Data Annotation. Driven both by the demand for high-quality extraction and the heterogeneity of the sources, the construction process is organized as a multi-stage pipeline (Figure 11) following the pipeline of P1(Chen et al., 2025b).

###### 2.3. Quality Control.

To ensure data reliability, we implement a rigorous multi-stage pipeline integrating automated model-based filtering with human verification. (1) OCR Correction. Texts parsed from complex layouts or low-quality scans are manually validated against original PDFs to rectify OCR artifacts. (2) Answer Cross-Validation. Three models — Gemini-2.5-Flash, Claude-3.7-Sonnet, and GPT-4o — independently extract answers from each question–solution pair. Consensus is established via majority vote (at least two models agree); non-consensus items are discarded. (3) Task Filtering. Problems requiring diagram generation or involving unverifiable open-ended answers (e.g., proofs) are removed. (4) Visual Consistency Check. We utilize Gemini-2.5-Flash to detect and filter out samples where the question text references a figure that is missing from the input. (5) Expert Review. A final consistency check is performed by Claude-3.7-Sonnet, followed by targeted manual refinement by experts. Through this filtering process, the dataset is refined from 13,432 to 8,033 items, yielding a high-fidelity, multi-modal bilingual corpus well-suited for RLVR.

### 3. Approach

###### 3.1. RL Formulation

We formulate the challenge of solving Physics Olympiad problems through the lens of reinforcement learning (RL) (Sutton et al., 1998) process. Formally, let M = (S, A, 𝑃, 𝑟) denote the underlying Markov Decision Process (MDP), where:

- • S represents the state space, corresponding to the model context, including the problem statement and all previously generated reasoning tokens.
- • A is the action space, which constitutes the discrete action space of possible tokens in the vocabulary.
- • 𝑃(𝑠′ | 𝑠, 𝑎) defines a (deterministic) state transition function, which appends the newly chosen action 𝑎 to the state 𝑠, resulting in an updated context 𝑠′.
- • 𝑟(𝑠, 𝑎) provides a scalar reward indicating the correctness and quality of the finalized reasoning trajectory.

The learning objective is to optimize the policy 𝜋𝜃 by maximizing the expected return:

###### ∑︁ 𝑇

𝑟(𝑠𝑡, 𝑎𝑡) , (1)

𝐽(𝜋𝜃) = 𝔼𝜏∼𝜋

𝜃

𝑡=0

where 𝜋𝜃(𝑎𝑡|𝑠𝑡) is the policy parameterized by model parameters 𝜃, and 𝜏 = (𝑠0, 𝑎0, . . . , 𝑠𝑇) denotes a trajectory sampled from 𝜋𝜃.

Policy Gradient. The policy gradient (Sutton et al., 1999) method optimizes 𝜋𝜃 by ascending the gradient of the expected return:

###### ∑︁ 𝑇

∇𝜃 log𝜋𝜃(𝑎𝑡 | 𝑠𝑡) 𝐴𝜋(𝑠𝑡, 𝑎𝑡) , (2)

∇𝜃𝐽(𝜋𝜃) = 𝔼𝜏∼𝜋

𝜃

𝑡=0

where 𝐴𝜋(𝑠𝑡, 𝑎𝑡) is the advantage function estimating the relative value of action 𝑎𝑡 in state 𝑠𝑡. We adopt this standard form and instantiate it via GSPO below.

Group Sequence Policy Optimization (GSPO). GSPO (Zheng et al., 2025a) elevates optimization from the token level (Shao et al., 2024; Yu et al., 2025c) to the sequence level, employing lengthnormalized sequence likelihood importance ratios:

###### ∑︁|𝑦𝑖|

1/|𝑦𝑖|

1 |𝑦𝑖|

𝜋𝜃(𝑦𝑖,𝑡|𝑥, 𝑦𝑖,<𝑡) 𝜋𝜃old(𝑦𝑖,𝑡|𝑥, 𝑦𝑖,<𝑡)

𝜋𝜃(𝑦𝑖|𝑥) 𝜋𝜃old(𝑦𝑖|𝑥)

log

= exp

, (3)

𝜌𝑖(𝜃) =

𝑡=1

where |𝑦𝑖| denotes the sequence length, and the 1/|𝑦𝑖| term implements length normalization to reduce variance. The corresponding advantage function is computed at the sequence level:

𝑅𝑖 − mean({𝑅𝑗}𝐺𝑖=1) std({𝑅𝑗}𝐺𝑖=1)

𝐴ˆGSPO𝑖 =

, (4)

with the objective function:

###### ∑︁𝐺

1

min 𝜌𝑖(𝜃)𝐴ˆGSPO𝑖 , clip(𝑠𝑖(𝜃), 1 − 𝜖, 1 + 𝜖)𝐴ˆGSPO𝑖 . (5)

𝐽GSPO(𝜃) = 𝔼𝑥∼D,{𝑦

𝑖}𝑖𝐺=1∼𝜋𝜃old (·|𝑥)

𝐺

𝑖=1

- 3.2. Instantiation We keep the same design of the reward and the verifier following P1 Series (Chen et al., 2025b).

Reward Design. We employ a binary reward scheme based on answer correctness, leveraging the fact that our physics dataset contains verifiable ground-truth outcomes:

𝑟 =

1, if the predicted answer matches the ground truth, 0, otherwise.

(6)

The we adopt the test-case-style reward aggregation, defining the final reward as:

###### ∑︁𝑁

1

𝑟𝑖, (7)

𝑅 =

𝑁

𝑖=1

where 𝑁 is the number of required sub-answers in the problem, and 𝑟𝑖 denotes the correctness indicator for the 𝑖-th sub-answer. Furthermore, we apply the system prompt as shown in Figure 5 to extract the multi-box answers.

System Prompt of Multi-box Style

Please answer the problem adhering to the following rules:

- 1. Please use LaTeX format to represent the variables and formulas used in the solution process and results.
- 2. Please put the final answer(s) in \boxed{}, note that the unit of the answer should not be included in \boxed{}.
- 3. If the problem requires multiple answers, list them in order, each in a separate \boxed{}.

- Figure 5 | System prompt design for P1-VL training.

Verifier Design. Physics problems frequently necessitate solutions in the form of complex symbolic expressions rather than scalar values, posing a challenge for standard verification metrics. We implement the hybrid verification framework that harmonizes rigorous symbolic logic with semantic model evaluation: (1) Rule-Based Verifier, we integrate symbolic computation libraries (SymPy (Meurer et al., 2017)) with heuristic strategies from math-verify (Kydlíček); (2) To capture correct answers that elude strict symbolic matching, we adopt the XVerify (Chen et al., 2025a) paradigm. We employ a specialized LLM (Qwen3-30B-A3B-Instruct) as a semantic judge.

- 3.3. Technical Design

- 3.3.1. Curriculum Training

Preliminary Difficulty Calculation and Filtering Strategy. The distribution of data difficulty and data quality plays a pivotal role in the convergence of RL training (Cui et al., 2025a; Zhang et al., 2025). To construct an effective training set, we adopt an inference-based metric to estimate the solvability of each instance. Let 𝐷(𝑥𝑖, 𝑦𝑖) denote the empirical pass rate:

###### ∑︁𝑁

1

𝕀(ˆ𝑦𝑘 = 𝑦𝑖), (8)

𝐷(𝑥𝑖, 𝑦𝑖) =

𝑁

𝑘=1

where ˆ𝑦𝑘 denotes the 𝑘-th response generated by the model. We utilize the Qwen3-VL-30B-A3B model to perform extensive rollouts (with 𝑁 = 72) on the training dataset. The correctness of inference rollouts is determined only by the rule-based verifier. Based on the distribution of 𝐷(𝑥𝑖, 𝑦𝑖), we apply a dual-end filtering strategy:

- 1. Pruning Trivial Samples (𝐷(𝑥𝑖, 𝑦𝑖) > 0.7): Samples that the model can already solve with high confidence contribute minimal gradient variance during optimization, accelerating entropy collapse of the policy model (Cui et al., 2025b). To improve training efficiency, we categorize samples with 𝐷 > 0.7 as "trivial" and remove them from the training set following Chen et al. (2025b).

- 2. Recovering Zero-Shot Failures (𝐷(𝑥𝑖, 𝑦𝑖) = 0.0): Samples where the pass rate is strictly 0.0 present a challenge; they represent either hard-core reasoning steps or malformed questions. Training on the malformed questions can lead to reward hacking or instability. Thus, we filter these malformed samples and introduce a recovery pipeline using Gemini-2.5-Flash. We prompt such a stronger model to (a) verify the alignment between the question and image, and (b) verify the completeness of the question, including physical factors or numerical values, (c) refine ambiguous problem descriptions. The unrecoverable tasks are finally excluded to maintain the overall quality. This process converts previously "unsolvable" noise into effective, clear RL tasks.

Curriculum RL Training. Directly training with RL on difficult problems for the VLMs could cause unstable policy gradients and low efficiency because of the low success rates, thus, we expand the difficulty of the tasks step by step. In the initial stage, we train the model with samples that exhibit a pass rate below an upper threshold of 0.7. By filtering out trivial instances (i.e., those with success rates 𝐷(𝑥𝑖, 𝑦𝑖) > 0.7) early in the process, we prevent the model from wasting computation on tasks it has already mastered. Subsequently, we further intensify the training challenge by lowering this inclusion threshold. This progressive tightening of the difficulty constraint effectively prunes moderately easy samples from the dataset, compelling the model to concentrate its optimization capacity exclusively on the harder tail of the distribution, where its performance is still lacking.

- As the curriculum shifts toward the harder tail of the distribution, static rollout configurations become restrictive, failing to provide the search depth required for these escalated challenges (Cui et al.,

2025b; Luo et al., 2025a). Thus, we follow P1(Chen et al., 2025b) to dynamically expand the exploration space, such as group size and generation window, in pace with the difficulty expansion, ensuring the search budget scales adequately to sustain long-term learnability.

- 3.3.2. Training Stabilization Mechanism

Mitigate Train-inference Mismatch Recent studies (Liu et al., 2025; Yao et al., 2025) have noticed that the train-inference engine difference is a key cause of instability in RL training. An empirical analysis is shown in Section 5.2. To formally understand this statement, Eq. 2 could be rewritten as,

###### ∑︁ 𝑇

∇𝜃 log𝜋𝑡𝑟𝑎𝑖𝑛𝜃 (𝑎𝑡 | 𝑠𝑡) 𝐴𝜋(𝑠𝑡, 𝑎𝑡) , (9)

∇𝜃𝐽(𝜋𝜃) = 𝔼𝜏∼𝜋𝑟𝑜𝑙𝑙𝑜𝑢𝑡

𝜃

𝑡=0

where 𝜋𝑟𝑜𝑙𝑙𝑜𝑢𝑡𝜃 denotes the policy used to generate trajectories during rollout, and 𝜋𝑡𝑟𝑎𝑖𝑛𝜃 denotes the policy evaluated during gradient computation. RL frameworks (Sheng et al., 2024; Zhu et al., 2025b)

often adopt different engines for rollout (e.g. vllm (Kwon et al., 2023) and SGLang (Zheng et al., 2024)) and training (e.g. FSDP (Merry et al., 2021) and Megatron (Shoeybi et al., 2019)), which introduce a mismatch between 𝜋𝑟𝑜𝑙𝑙𝑜𝑢𝑡𝜃 and 𝜋𝑡𝑟𝑎𝑖𝑛𝜃 due to differences in numerical precision, computation optimization strategies, and kernel implementations, leading to biased gradient estimates and training instability: 𝜋𝑟𝑜𝑙𝑙𝑜𝑢𝑡𝜃 (𝑎|𝑠) ≠ 𝜋𝑡𝑟𝑎𝑖𝑛𝜃 (𝑎|𝑠).

Masked Importance Sampling. To mitigate the instability caused by the training-inference mismatch, we adopt Sequence-level Masked Importance Sampling (Seq-MIS) (Liu et al., 2025). Unlike tokenlevel clipping methods, Seq-MIS employs a "hard trust region" by rejecting entire out-of-distribution trajectories. However, the cumulative importance weight 𝜌(𝜏) scales exponentially with sequence length 𝑇, making it difficult to set a consistent rejection threshold 𝐶 for tasks with varying lengths. To address this, we adopt the Geometric Mean of the importance weights to normalize the metric across

different lengths. The gradient estimator is formulated as:





∑︁𝑇

𝕀 𝜌(𝜏)1/𝑇 ≤ 𝐶

∇𝜃 log𝜋𝜃(𝑎𝑡|𝑠𝑡)𝐴𝜋(𝑠𝑡, 𝑎𝑡)

, (10)

∇𝜃𝐽Geo-MIS(𝜃) = 𝔼𝜏∼𝜋old

·𝜌(𝜏) ·

𝑡=0

 

 

Geo-Mask

where 𝜏 = (𝑠0, 𝑎0, . . . , 𝑠𝑇, 𝑎𝑇) represents the sampled trajectory, and 𝑇 is the sequence length. The sequence-level importance weight is defined as 𝜌(𝜏) = 𝑇𝑡=0 𝜋

trainer

𝜃 (𝑎𝑡|𝑠𝑡)

𝜋rollout𝜃 (𝑎𝑡|𝑠𝑡) . 𝕀(·) denotes the indicator function, rejecting trajectories whose geometric mean of importance weights exceeds the threshold 𝐶.

- 3.4. Training Dynamics Implementation. For the implementation of P1-VL training pipeline, we adopt the VERL (Sheng et al.,

- 2024) framework, which is an efficient VLMs post-training framework offering the RL implementation in Megatron. We adopt Qwen3-VL-30B-A3B-Thinking and Qwen3-VL-235B-A22B-Thinking as starting points of models. To adaptively adjust the aforementioned learnability during training, we periodically resume training from the previous checkpoint with updated configurations.

Image Processing For questions containing images, the special token <image> is inserted at the corresponding position within the text. To facilitate unified RL training across modalities, we standardize the input format by padding all text-only samples with a blank image of resolution 448 × 448, and appending the <image> token to the end of the question. Vision-language models (VLMs), such as Qwen3-VL (Bai et al., 2025b) and Intern3.5-VL (Wang et al., 2025), typically consist of an image encoder, a language model, and a projection layer that bridges the two. Since our training data is not specifically curated for image alignment, we freeze the parameters of both the vision encoder and the projection layer during RL training. The analysis of training data composition is shown in Section

- 5.3. Furthermore, we empirically observe that padding images has a negligible impact on text-only performance. Therefore, we omit blank images during evaluation.

Settings of Curriculum Training Phrases. We detail the stage-wise hyperparameter settings of the P1-VL post-training process in Table 2. Following our curriculum framework, we progressively amplify the training complexity by scaling the data difficulty (metric defined in Eq. 8) and expanding the exploration space via larger group sizes and extended generation windows. To bridge the train-inference gap, we consistently apply Masked Importance Sampling—a correction technique for off-policy stability—across all stages. Notably, our optimization backbone is built upon the GSPO algorithm to ensure stable convergence for MoE architectures. Furthermore, we exclusively employ rule-based verifiers during training to mitigate reward hacking.

Analysis. Figure 6 illustrates the training dynamics across the curriculum RL process. A key observation is the gradual increase in response length, indicating that the model is developing the capacity for deeper reasoning to tackle complex problems. To accommodate this evolving capability, we strategically expand the difficulty and meanwhile expand exploration space by increasing the generation window and group size at each stage transition. Quantitatively, validation results on HiPhO demonstrate a steady performance improvement throughout training, underscoring both the effectiveness and the stability of our algorithm. A detailed analysis of the effect of expansion is shown in Section 5.4.

- Table 2 | Configuration of different phrases in P1-VL training.

P1-VL-30B-A3B P1-VL-235B-A22B

Model

Stage 1 Stage 2 Stage 3 Stage 1 Stage 2 Stage 3

Group size 8 8 16 8 8 16 Difficulty (0.0,0.7] (0.0,0.5] [0.0,0.5] (0.0,0.7] (0.0,0.5] [0.0.0.5] Generation Window 40k 60k 80k 48k 54k 54k Learning Rate 1 × 10−6 1 × 10−6 Algorithm GSPO w. MIS GSPO w. MIS Verifier Choice Rule-based Verifier Only Rule-based Verifier Only Rollout Batch Size 2048 1024 Update Batch Size 256 256

- 3.5. Agentic Augmentation

At inference time, we enhance test-time reasoning with a unified agentic augmentation framework, PhysicsMinions (Yu et al., 2025b), a pioneering coevolutionary multi-agent system for multimodal scientific problem solving. The system is organized into three tightly coupled studios: the Visual Studio, the Logic Studio, and the Review Studio. The Visual Studio performs structured perception and converts raw visual evidence into symbolic representations. The Logic Studio generates and iteratively refines candidate solutions through solver and introspector collaboration. The Review Studio validates solutions using both domain-specific and general verification. Together, these studios form a closed critique and refinement loop that enables scalable test-time reasoning and improves robustness on complex scientific tasks.

A key distinction from the earlier text-only P1 deployment of PhysicsMinions (Chen et al., 2025b) is that P1-VL fully activates the Visual Studio to process diagrams, plots, and other visual inputs. By grounding reasoning in structured visual representations, the system can directly incorporate visual evidence into downstream symbolic reasoning, enabling effective handling of multimodal Olympiad-style problems.

Beyond multimodal perception, we extend the agent pipeline with a domain-adaptive mechanism that generalizes the framework to heterogeneous scientific disciplines. The system automatically selects discipline-specific solver prompts and domain verifiers according to the detected problem type, such as physics, chemistry, or biology, while all remaining agent components and interaction protocols are shared. On the physics-only HiPhO benchmark (Yu et al., 2025a), the system employs the Physics-Verifier in a setting consistent with prior physics-focused evaluations. On the multidisciplinary FrontierScience-Olympiad benchmark (Wang et al., 2026), the same pipeline dynamically routes problems to the appropriate domain verifier and solver configuration. These experiments demonstrate that combining visual grounding with domain-adaptive agentic reasoning supports effective performance across physics, chemistry, and biology Olympiad-level problems.

- 4. Experiment

- 4.1. Experimental Setup

Test Dataset. To rigorously evaluate performance on advanced physical reasoning tasks, we introduce the HiPhO benchmark (Yu et al., 2025a), the first dataset dedicated exclusively to High School Physics Olympiads. HiPhO aggregates 13 major exams administered between 2024 and 2025, spanning seven prestigious international and regional competition series: IPhO, APhO, EuPhO, NBPhO, PanPhO, PanMechanics, and F=MA. The inclusion criteria prioritized both global influence and the availability

###### P1-VL-30B-A3B Response Lengths

###### P1-VL-30B-A3B HiPhO Overall Points

###### AverageResponseLength

20000

380

###### OverallPoints

18000

360

16000

340

14000

12000

320

Stage 1 Stage 2 Stage 3

Stage 1 Stage 2 Stage 3

0 200 400 600 800 1000 1200 1400

0 200 400 600 800 1000 1200 1400

Training Steps

Training Steps

###### P1-VL-235B-A22B Response Lengths

P1-VL-235B-A22B HiPhO Overall Points

###### AverageResponseLength

| |Stage 1| | |Stage 2| | |Stage 3| | | |
|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |

440

20000

430

OverallPoints

18000

420

16000

410

14000

400

12000

390

10000

Stage 1 Stage 2 Stage 3

380

0 200 400 600 800 1000 1200 1400

0 200 400 600 800 1000 1200 1400

Training Steps

Training Steps

- Figure 6 | The training dynamics of P1-VL curriculum RL. The upper side shows the dynamics of P1-VL-30B-A3B, and the lower shows those of P1-VL-235B-A22B. The training is set in three stages with curriculum difficulty expansion, group size expansion, and generation window expansion. The average response length represents the generation length during training, and the overall points refer to the sum score of all the exams in HiPhO with model-based verifier (Qwen-3-30B-Instruct-2507).

of granular human performance data to facilitate direct human-AI comparison1. Comparison Models. We compared against 39 representative models, including 13 closed-source and 26 open-source models, selected to reflect the current frontier of LLMs and VLMs :

- • Closed-source models: Gemini-3-Pro (Deepmind), GPT-5.2 (OpenAI, c), GPT-5 (OpenAI, b), o3 (OpenAI, d), o4-mini (high) (OpenAI, d), o4-mini (OpenAI, d), GPT-4o (OpenAI, a), Gemini-2.5Pro (Comanici et al., 2025), Gemini-2.5-Flash-Thinking (Comanici et al., 2025), Grok-4 (xAI), Claude-4-Sonnet-Thinking (Anthropic), Claude-4-Sonnet (Anthropic), Mistral-Medium-3 (Mistral).
- • Open-source models: DeepSeek-V3.2-Thinking (DeepSeek-AI et al., 2025), Qwen3-VL Series (Bai et al., 2025b), Intern-S1 (Bai et al., 2025a), InternVL3 Series (Zhu et al., 2025a), Qwen2.5-VL Series (Bai et al., 2025c), GLM-4.5V (Team et al., 2025a), DeepSeek-VL2 (Wu et al., 2024), LLaMA4Scout-17B (Meta), Phi-4-multimodal (Abouelenin et al., 2025), P1 Series (Chen et al., 2025b), GPT-OSS-120B (Agarwal et al., 2025), Kimi-K2-Thinking (Team et al., 2025b), Kimi-K2-Instruct (Team et al., 2025b), DeepSeek-R1 (Guo et al., 2025), DeepSeek-V3 (Liu et al., 2024), Qwen3 Series (Yang et al., 2025).

Model Configurations. All models were evaluated under a standardized inference setup following the HiPhO protocol. The temperature was fixed at 0.6, and the maximum token limit was set as large as permitted by each model. For every problem, we conducted 8 independent inference runs and computed the average score per problem. These averages were then aggregated across problems to yield the final exam score for each Olympiad.

1Notable competitions such as CPhO, USAPhO, and APhO-2024 were excluded due to the unavailability of complete official contestant score distributions.

Evaluation Method. Adhering to the HiPhO benchmark protocol (Yu et al., 2025a), we employ Gemini-2.5-Flash as the automated evaluator. The evaluation framework mirrors official Olympiad grading by integrating both answer-level verification and step-level reasoning analysis. For each problem, the final score is determined as the maximum of the answer-based and step-based evaluations. This mechanism ensures consistency with human adjudication: a correct final answer gains full credit, while rigorous intermediate derivations earn partial points even if the final result is incorrect. Consequently, we report the cumulative exam score rather than standard accuracy, facilitating a direct comparison between model performance and official human medal thresholds.

###### 4.2. Evaluation on Physics Olympiads

Superior Single-Model Performance. Evaluation results are presented in Table 3, verifying the effectiveness of our RL training strategy. As noted above, the P1-VL model series is trained purely through reinforcement learning.

- • P1-VL-235B-A22B ranks 3rd among all evaluated models, following only Gemini-3-Pro and GPT-

5.2. With 12 gold and 1 silver medals (average score: 39.3), it surpasses the closed-source models such as Gemini-2.5-Pro (37.7), GPT-5 (37.4), and Grok-4 (35.8). Notably, the standalone P1-VL outperforms the agent-augmented baseline (P1-235B-A22B + PhysicsMinions, 38.4), proving that intrinsic visual reasoning capabilities can surpass purely agentic workflows in complex physical scenarios, which incorporates image caption information. It shows that our models could effectively perceive the image information and bridge it to the scientific reasoning (case study for Text+Variable Figure is shown in Appendix A.2). Compared with its base model (Qwen3-VL-235B-A22B-Thinking, 33.9), it improves 5.4 points on average.

- • P1-VL-30B-A3B achieves 9 gold and 4 silver medals with an average score of 35.0. It ranks third among open-source models, behind only DeepSeek-V3.2-Thinking and P1-235B-A22B. Despite its smaller size, it outperforms larger baselines such as Qwen3-VL-235B-A22B-Thinking (33.9) and Qwen3-235B-A22B-Thinking-2507 (33.5), demonstrating significant parameter efficiency. It also outperforms some closed-source models such as o4-mini and Gemini-2.5-Flash-Thinking, showing excellent performance on scientific reasoning.

Agentic Augmentation. With the agentic augmentation of PhysicsMinions, the average score of P1-VL-235B-A22B improves from 39.3 to 40.9, propelling it to the 2nd rank globally and surpassing the closed-source GPT-5.2 (40.6). This combined system establishes new state-of-the-art performance on three physics Olympiads, including PanPhO 2025 (66.5 vs 66.3), PanPhO 2024 (83.3 vs 82.5), and PanMechanics 2024 (84.8 vs 82.3). These gains underscore the efficacy of the "model + system" paradigm, demonstrating how multi-agent collaboration significantly amplifies complex scientific reasoning capabilities.

- Table 3 | Evaluation results on the HiPhO benchmark (13 physics Olympiads from 2024–2025) using the exam score metric. Gold, Silver and Bronze indicate scores above the respective thresholds. Models are ranked by average scores; bold is the highest score, and underline is the second highest. Here, only the theoretical parts of exams are used, hence Full Mark (Model) ≤ Full Mark (Human).

Physics Olympiad IPhO APhO EuPhO NBPhO PanPhO PanMechanics F=MA Avg. Medal Year 2025 2024 2025 2025 2024 2025 2024 2025 2024 2025 2024 2025 2024 Table

Full Mark (Human) 30.0 30.0 30.0 30.0 30.0 72.0 72.0 100.0 100.0 100.0 100.0 25.0 25.0 57.2 Full Mark (Model) 29.4 29.3 30.0 29.0 28.0 43.5 50.0 100.0 98.0 100.0 100.0 25.0 25.0 52.9 Top-1 Score (Human) 29.2 29.4 30.0 27.0 30.0 53.2 40.8 81.0 66.5 62.0 51.0 25.0 24.0 42.2 Top-1 Score (Compared Models) 25.2 25.9 28.4 21.0 23.9 37.1 43.8 66.3 82.5 86.0 82.3 23.8 23.9 43.9 Gold Medal 19.7 20.8 23.3 16.5 20.4 28.6 26.5 41.5 52.0 52.0 51.0 15.0 14.0 29.3 Silver Medal 12.1 11.1 18.7 9.8 14.2 20.1 19.4 28.5 37.5 36.0 26.0 11.0 12.0 19.7 Bronze Medal 7.2 3.6 13.1 5.8 8.9 15.2 13.5 14.5 16.0 20.0 12.0 9.0 10.0 11.4

Gemini-3-Pro (high) 25.2 25.2 28.3 21.0 23.9 37.1 43.8 66.3 76.2 86.0 74.8 23.8 23.9 42.7 13 0 0 P1-VL-235B-A22B+PhysicsMinions 22.1 25.8 27.1 13.7 22.9 31.8 32.6 66.5 83.3 76.0 84.8 23.1 21.8 40.9 12 1 0 GPT-5.2 (high) 22.3 25.5 28.4 15.7 23.9 35.8 37.1 59.8 82.5 67.6 82.3 23.4 23.5 40.6 12 1 0 P1-VL-235B-A22B 21.0 25.3 26.9 10.7 21.5 32.5 31.9 61.7 79.8 73.3 82.0 22.5 21.9 39.3 12 1 0 P1-235B-A22B+PhysicsMinions 23.2 25.2 28.0 12.4 23.5 31.9 35.4 57.7 67.0 77.5 74.8 21.5 20.5 38.4 12 1 0 Gemini-2.5-Pro 22.7 25.9 27.9 14.9 21.8 32.3 35.9 60.3 64.1 69.5 70.2 22.8 22.0 37.7 12 1 0 GPT-5 22.3 20.2 27.0 10.3 21.7 32.9 32.8 55.9 69.8 69.4 79.0 22.4 22.4 37.4 11 2 0 DeepSeek-V3.2-Thinking 20.2 24.9 26.1 6.7 18.9 33.3 29.7 57.3 64.9 78.6 78.0 21.5 19.9 36.9 11 1 1 P1-235B-A22B 21.2 24.7 27.4 10.8 23.0 31.8 28.4 54.7 56.7 74.7 72.9 20.9 19.4 35.9 12 1 0 Grok-4 18.7 23.5 25.0 11.5 20.5 25.8 29.3 45.0 75.4 72.1 78.6 19.8 19.8 35.8 10 3 0

- o3 15.7 23.7 25.9 11.4 21.6 34.1 33.5 47.3 55.9 71.4 75.6 22.0 20.6 35.3 11 2 0 P1-VL-30B-A3B 17.9 22.5 24.6 10.6 19.0 30.2 24.6 50.8 58.6 79.1 75.5 20.8 20.3 35.0 9 4 0 Qwen3-VL-235B-A22B-Thinking 19.8 23.4 25.4 10.8 19.0 29.2 25.9 48.3 54.0 64.3 79.9 21.4 19.9 33.9 10 3 0 Qwen3-235B-A22B-Thinking-2507 17.1 23.0 26.2 10.9 20.4 33.6 28.1 44.7 51.8 69.1 72.9 18.5 18.9 33.5 10 3 0 P1-30B-A3B 18.5 22.3 25.4 7.4 18.8 29.2 24.0 47.0 51.4 69.5 69.1 19.6 20.0 32.5 8 4 1 GPT-OSS-120B (high) 20.5 22.1 25.8 12.8 23.0 32.4 32.4 48.8 52.4 54.7 56.1 20.5 19.2 32.4 12 1 0

- o4-mini (high) 16.0 23.7 22.9 12.0 20.1 27.4 29.8 41.4 50.9 69.1 67.3 18.6 18.8 32.2 6 7 0 Gemini-2.5-Flash-Thinking 20.2 23.9 27.4 13.2 21.9 29.0 29.3 44.6 54.9 60.5 55.9 17.8 19.1 32.1 12 1 0 DeepSeek-R1 18.5 24.6 25.4 10.8 21.4 26.3 20.5 42.2 47.4 65.4 72.5 18.3 18.5 31.7 8 5 0

- o4-mini 15.4 22.9 22.8 10.1 20.9 26.9 27.3 39.4 47.1 64.2 62.5 18.6 18.5 30.5 7 6 0 Kimi-K2-Thinking* 19.1 22.8 22.4 5.2 20.9 26.0 20.6 52.5 51.3 47.3 64.8 20.6 19.7 30.2 6 6 0 Claude-4-Sonnet-Thinking 19.0 22.0 24.8 9.7 20.5 28.1 25.6 43.1 39.3 57.4 61.8 19.2 20.1 30.0 8 4 1 Qwen3-30B-A3B-Thinking-2507 15.6 19.7 23.5 7.4 16.4 28.6 22.2 40.5 43.8 67.7 66.5 18.3 18.0 29.9 6 6 1 Qwen3-VL-30B-A3B-Thinking 15.7 19.3 25.3 11.7 20.2 29.7 28.3 41.6 45.9 57.6 51.4 18.9 20.7 29.7 8 5 0 Qwen3-32B 15.7 19.3 23.9 9.8 21.2 28.9 24.1 36.6 41.8 67.0 59.2 18.9 16.6 29.5 7 6 0 Kimi-K2-Instruct 16.5 19.8 24.2 11.0 16.9 26.5 26.2 35.9 41.8 65.9 58.9 16.0 18.2 29.1 5 8 0 GPT-OSS-120B 16.9 21.4 22.8 9.1 19.9 26.0 25.8 37.4 41.8 57.1 59.7 17.8 17.6 28.7 5 7 1 Intern-S1 15.9 14.2 21.7 9.0 16.6 23.0 20.5 41.1 50.3 60.4 57.4 18.4 19.5 28.3 4 8 1 Claude-4-Sonnet 15.7 19.2 22.8 9.5 16.5 27.5 21.3 40.4 43.3 46.5 48.5 16.8 16.5 26.5 2 10 1 DeepSeek-V3 13.6 16.4 22.1 7.1 17.2 21.1 17.3 37.2 35.0 48.4 46.5 14.1 15.6 24.0 1 9 3 Mistral-Medium-3 14.2 14.1 19.9 8.5 12.2 20.4 19.6 30.8 28.6 32.9 36.1 13.9 14.1 20.4 1 8 4 InternVL3-78B-Instruct 12.9 12.5 17.7 7.5 15.2 22.3 22.5 26.2 27.4 21.1 27.1 12.0 13.0 18.3 0 8 5 GLM-4.5V 11.9 4.4 16.2 8.7 14.1 19.5 14.0 18.5 16.0 47.8 39.0 13.0 13.8 18.2 0 4 9 LLaMA4-Scout-17B 9.7 9.5 13.1 5.4 10.4 22.8 12.8 26.6 24.1 35.4 34.5 8.1 6.4 16.8 0 2 7 GPT-4o 10.2 9.4 15.1 6.8 9.2 16.4 11.7 27.8 22.8 28.2 26.5 15.0 10.9 16.2 1 1 10 Qwen3-8B 10.6 12.7 11.5 7.1 11.9 20.1 17.3 26.3 22.3 21.8 22.8 10.8 10.0 15.8 0 2 10 Qwen2.5-VL-32B-Instruct 9.9 8.2 16.5 6.9 10.0 15.3 14.4 22.5 22.4 28.1 29.9 7.6 4.6 15.1 0 1 10 Qwen2.5-VL-72B-Instruct 10.6 7.2 13.6 6.1 8.1 13.3 11.4 26.8 18.2 24.0 28.5 13.5 9.8 14.7 0 2 7 InternVL3-38B-Instruct 8.9 7.8 12.3 6.1 8.3 14.0 10.6 24.1 20.4 27.5 24.8 8.2 6.8 13.8 0 0 7 InternVL3-9B-Instruct 4.7 3.7 7.2 4.2 4.1 9.4 6.0 12.4 11.0 11.6 16.3 6.4 6.2 7.9 0 0 2 Qwen2.5-VL-7B-Instruct 3.5 2.5 5.7 4.4 3.6 7.3 5.5 14.7 7.6 9.8 11.5 4.4 3.5 6.5 0 0 1 Phi-4-multimodal 2.0 1.6 4.2 3.6 3.6 5.0 4.5 8.3 9.0 10.0 10.1 4.4 5.0 5.5 0 0 0 DeepSeek-VL2 1.8 0.5 2.5 3.4 3.4 5.0 3.2 5.6 4.8 7.3 6.4 5.0 3.9 4.0 0 0 0

###### * For Kimi-K2-Thinking, the inference timeout is set to 2 hours, and 9.58% of cases exceed this limit.

### 5. Discussion

###### 5.1. Generalizability of P1-VL

We conduct specialized post-training to enhance physics problem-solving; however, a critical question remains: Does this domain-specific optimization compromise or catalyze general STEM reasoning?

To investigate the model’s generalization capabilities, we evaluate P1-VL on the open-source benchmark FrontierScience-Olympiad (Wang et al., 2026). As detailed in Table 4, the results demonstrate robust positive transfer: both P1-VL-235B-A22B and P1-VL-30B-A3B achieve significant gains over their base counterparts across all three scientific domains, yielding a total score improvement of 8.0 and 9.1, respectively. Remarkably, even on this text-only benchmark, the multimodal P1-VL-235B-A22B outperforms its text-only sibling (P1-235B-A22B) by a margin of 2.3 points. Furthermore, when augmented with the PhysicsMinions agent framework, P1-VL-235B-A22B+PhysicsMinions attains a total score of 67.1, securing state-of-the-art performance among all evaluated open-source models.

- Table 4 | Evaluation results on the FrontierScience-Olympiad benchmark. The state-of-the-art performance of all the models is marked in bold, and that of open-source models is underlined. ∗ refers to the evaluation results reported in Wang et al. (2026).

Model Biology/10 Chemistry/40 Physics/50 Total/100

GPT-5.2∗ 43.5 89.0 74.3 77.1 Gemini-3-Pro∗ 41.0 85.6 75.5 76.1 Claude-Opus-4.5∗ 24.0 81.8 72.5 71.4 GPT-5.1∗ 33.5 82.4 67.4 70.0 GPT-5∗ 35.5 81.5 67.0 69.7 P1-VL-235B-A22B+PhysicsMinions 26.3 77.2 67.3 67.1 Grok-4∗ 33.0 73.2 67.2 66.2 DeepSeek-V3.2-Thinking 26.3 74.1 67.3 65.9 P1-235B-A22B+PhysicsMinions 30.0 71.0 68.0 65.4 Kimi-K2-Thinking 20.0 76.6 65.0 65.1 GLM-4.7 20.0 70.6 69.5 65.0 P1-VL-235B-A22B 30.0 71.3 65.5 64.3

- o3∗ 30.0 76.1 59.0 62.9 P1-235B-A22B 22.5 67.2 65.8 62.0
- o4-mini∗ 41.5 71.5 57.8 61.7 GPT-OSS-120B (high) 38.8 63.4 61.5 60.0 Qwen3-VL-235B-A22B-Thinking 26.3 61.9 57.8 56.3 Qwen3-235B-A22B-Thinking-2507 26.3 58.1 57.3 54.5 P1-30B-A3B 15.0 61.9 56.3 54.4 P1-VL-30B-A3B 20.0 58.8 54.0 52.5 Qwen3-VL-30B-A3B-Thinking 18.8 49.4 43.5 43.4 Qwen3-30B-A3B-Thinking-2507 10.0 47.8 45.3 42.8

- o1∗ 20.0 50.9 40.3 42.5 GPT-4o∗ 3.0 12.4 14.1 12.3

Besides, we compare the P1-VL series with their respective base models across diverse benchmarks:

• Text Benchmarks: ten STEM-oriented evaluation datasets (AIME24, AIME25, HMMT-Nov, HMMT-Feb, Brumo, CMICC (Balunović et al., 2025), IMO-AnswerBench (Luong et al., 2025), AMOBench (An et al., 2025), BeyondAIME (ByteDance-Seed, 2025), GPQA (Rein et al., 2024)), and a general reasoning task (LiveBench (White et al., 2024)).

###### P1-VL-30B-A3B VS Qwen3-VL-30B-A3B

100

Qwen3-VL-30B-A3B-Thinking P1-VL-30B-A3B

90.4 87.9

90.0

89.2

85.4

83.7

83.8

85

80.8

79.4

79.2

79.1 76.5

74.8

73.6

73.4 73.1 71.3

73.3

72.7

70.0

70

Score (%)

65.9

65.3

63.4 64.8

63.8

62.3 61.4

60.3

55

44.5

37.0

40

25

13.4

12.3

10

AIME24 AIME25HMMT-FebHMMT-NovIMO-AnswerbenchAMOBenchBeyondAIME Brumo CMICC GPQALiveBench HLE MMMUMMMU-ProEMMA-MiniMathVista-Mini

###### P1-VL-235B-A22B VS Qwen3-VL-235B-A22B

Qwen3-VL-235B-A22B-Thinking P1-VL-235B-A22B

100

93.8 92.1

93.3 90.8

93.3

90.0

88.3

84.2

83.9

83.3

83.1 81.4 79.9

82.6

81.6

85

77.1 79.4

78.0

77.2

72.9

70.2 71.3

70.6

70.6

69.7 69.6

68.5

70

Score (%)

62.3

55

47.5

39.0

40

25

15.9

13.9

10

AIME24 AIME25HMMT-FebHMMT-NovIMO-AnswerbenchAMOBenchBeyondAIME Brumo CMICC GPQALiveBench HLE MMMUMMMU-ProEMMA-MiniMathVista-Mini

- Figure 7 | Evaluation of P1-VL models against their respective base models on out-of-domain benchmarks. The evaluation suite encompasses both text-only and multi-modal tasks, spanning mathematical reasoning and broader STEM disciplines.

• Multi-modal Benchmarks: five STEM-oriented benchmarks, including HLE (Phan et al., 2025)), MMMU (Yue et al., 2024a), MMMU-Pro (Yue et al., 2024b), EMMA-Mini (Hao et al., 2025), and MathVista-Mini (Lu et al., 2024)2.

Figure 7 summarizes the comparative results, illustrating that our models consistently surpass their base counterparts across both text-only and multi-modal benchmarks. For example, in the domain of advanced mathematics, P1-VL-235B-A22B and P1-VL-30B-A3B outperform their respective baselines on AMOBench by margins of 8.5 and 7.5 points. This pattern confirms that P1-VLs not only retain but actively enhance reasoning capabilities beyond the target physics domain, extending even to rigorous challenges such as IMO-AnswerBench and AMOBench. The advantage is not limited to text-based reasoning but extends significantly to visual perception. Taking the multi-image reasoning task EMMA-Mini as a prime example, P1-VL-235B-A22B and P1-VL-30B-A3B achieve a substantial gain of 1.7 and 3.4 points over their baseline. This result specifically highlights the model’s enhanced capacity to synthesize information across multiple images and little catastrophic forgetting issue (Luo et al., 2025b) during RL training, validating the effectiveness of our vision-language alignment beyond standard physics problems.

2For MathVista-Mini and MMMU, we adopt the evaluation scripts posted on the Qwen3-VL repository, and for MMMU-Pro we adopt the setting in MMMU repository.

###### 5.2. Train-inference Mismatch

During the RL post-training of Qwen3-VL-MoE models on our physics data, we observe a catastrophic training collapse as illustrated in Figure 8. This instability aligns with the traininginference mismatch phenomenon characterized in recent analyses (Liu et al., 2025; Yao et al., 2025), where discrepancies between the inference engine and training framework lead to divergent policy updates. While implementing Truncated Importance Sampling (TIS) (Yao et al., 2025) postpones the onset of this collapse, it still encounters an instability in our experiments. In contrast, adopting Sequence-Level Masked Importance Sampling (Seq-MIS) (Liu et al., 2025) could stabilize the training process. By strictly rejecting out-of-distribution samples rather than merely clipping their weights, SeqMIS could maintain more rigorous policy alignment; thus, we adopt Seq-MIS throughout our P1-VL RL post-training.

0.65

0.55

Rewards

0.45

0.35

A3B + MIS A3B + TIS A3B

0.25

0.15

32 88 144 200 256 312 368

Training Steps

Figure 8 | Driven by training-inference mismatch, the RL training of Qwen3-VL-30B-A3B-Thinking MoE model suffers from severe collapse. The implementation of Masked Importance Sampling (MIS) proves effective in stabilizing the training.

###### 5.3. Effect of Training Data Composition

We also empirically analyze the impact of training data composition. Figure 9 illustrates the performance of Qwen3-VL-4B-Thinking (using XVerifier) when trained exclusively on questions containing images. Crucially, we observe that incorporating text-only data (padded with blank images) induces no negative transfer; rather, it yields superior performance almost all the time. It demonstrates the feasibility and superiority of using mixed multi-modal data. Consequently, we train our P1-VL models using a combination of both the text-only and image-text data.

300

HiPhO Overall Points

270

240

210

w image-text data w mixed data

180

150

80 160 240 320 400 480 560

Training Steps

Figure 9 | Comparison of Qwen3-VL-4B-Thinking training performance on the image-text data versus the mixed data (text-only + image-text).

###### 5.4. Effect of Curriculum Training

Figure 10 illustrates the impact of curriculum training on model evolution. Each line includes a period of observation after the parameters are fixed. It shows that in the absence of difficulty expansion, the response length fluctuates stagnantly across Stages 1 and 2, resulting in negligible performance gains. Conversely, introducing the curriculum strategy yields a marked increase in both average response length and reasoning accuracy. These results underscore the critical role of our curriculum RL framework in driving deep reasoning.

###### P1-VL-235B-A22B Response Lengths

P1-VL-235B-A22B HiPhO Overall Results

| |Stage 1| | |Stage 2| | |Stage 3| | | |
|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |

440

###### AverageResponseLength

20000

430

18000

OverallPoints

420

16000

410

14000

400

12000

390

10000

Stage 1 Stage 2 Stage 3

380

0 200 400 600 800 1000 1200 1400

0 200 400 600 800 1000 1200 1400

Training Steps

Training Steps

Figure 10 | Training dynamics of RL training with and without curriculum difficulty expansion.

### 6. Conclusion

In this work, we presented P1-VL, the first open-source Vision-Language Model family capable of achieving Olympiad-level proficiency in physics. By addressing the critical visual-logical gap inherent in physical problem solving, P1-VL overcomes the limitations of text-only reasoning. We proposed to improve the reasoning capacity by Curriculum RL Training to internalize robust reasoning patterns and Agentic Augmentation to enable rigorous test-time reflection. The evaluation results on the HiPhO benchmark, where P1-VL-235B-A22B secured 12 gold medals and demonstrated superior generalization across broader scientific disciplines, underscore the potential of P1-VLs in modeling complex scientific reasoning systems. We posit that the ability to synthesize visual constraints with causal logic is essential for the development of reliable world models and embodied AI. By releasing P1-VL to the research community, we aim to accelerate the transition from symbolic manipulation to genuine machine scientific discovery, paving the way for AI systems that can truly understand and navigate the physical world.

### 7. Acknowlegement

This work is supported by Shanghai AI Laboratory and a locally commissioned task from the Shanghai Municipal Government. We would like to extend our special thanks to the developers and maintainers of the following open-source projects, which have been critical to the implementation of this work. This includes Qwen3-VL (Bai et al., 2025b) and Qwen (Yang et al., 2025), which provided the foundational base models for our research; slime (Zhu et al., 2025b), whose innovative framework enabled efficient reinforcement learning in our training pipeline; and verl (Sheng et al., 2024), which offered a versatile reinforcement learning framework to support model training. We also thank sglang (Zheng et al., 2024) for its efficient infrastructure for LLM and VLM serving and inference, and Megatron-LM (Shoeybi et al., 2019) for providing the large-scale model training framework.

### References

Abdelrahman Abouelenin, Atabak Ashfaq, Adam Atkinson, Hany Awadalla, Nguyen Bach, Jianmin Bao, Alon Benhaim, Martin Cai, Vishrav Chaudhary, Congcong Chen, et al. Phi-4-mini technical report: Compact yet powerful multimodal language models via mixture-of-loras. arXiv preprint arXiv:2503.01743, 2025.

Sandhini Agarwal, Lama Ahmad, Jason Ai, Sam Altman, Andy Applebaum, Edwin Arbus, Rahul K

Arora, Yu Bai, Bowen Baker, Haiming Bao, et al. gpt-oss-120b & gpt-oss-20b model card. arXiv preprint arXiv:2508.10925, 2025.

Shengnan An, Xunliang Cai, Xuezhi Cao, Xiaoyu Li, Yehao Lin, Junlin Liu, Xinxuan Lv, Dan Ma, Xuanlin Wang, Ziwen Wang, and Shuang Zhou. Amo-bench: Large language models still struggle in high school math competitions, 2025. URL https://arxiv.org/abs/2510.26768.

#### Anthropic. Claude 3.7 sonnet system card. URL https://www.anthropic.com/news/ visible-extended-thinking.

Lei Bai, Zhongrui Cai, Maosong Cao, Weihan Cao, Chiyu Chen, Haojiong Chen, Kai Chen, Pengcheng Chen, Ying Chen, Yongkang Chen, Yu Cheng, Yu Cheng, Pei Chu, Tao Chu, Erfei Cui, Ganqu Cui, Long Cui, Ziyun Cui, Nianchen Deng, Ning Ding, Nanqin Dong, Peijie Dong, Shihan Dou, Sinan Du, Haodong Duan, Caihua Fan, Ben Gao, Changjiang Gao, Jianfei Gao, Songyang Gao, Yang Gao, Zhangwei Gao, Jiaye Ge, Qiming Ge, Lixin Gu, Yuzhe Gu, Aijia Guo, Qipeng Guo, Xu Guo, Conghui He, Junjun He, Yili Hong, Siyuan Hou, Caiyu Hu, Hanglei Hu, Jucheng Hu, Ming Hu, Zhouqi Hua, Haian Huang, Junhao Huang, Xu Huang, Zixian Huang, Zhe Jiang, Lingkai Kong, Linyang Li, Peiji Li, Pengze Li, Shuaibin Li, Tianbin Li, Wei Li, Yuqiang Li, Dahua Lin, Junyao Lin, Tianyi Lin, Zhishan Lin, Hongwei Liu, Jiangning Liu, Jiyao Liu, Junnan Liu, Kai Liu, Kaiwen Liu, Kuikun Liu, Shichun Liu, Shudong Liu, Wei Liu, Xinyao Liu, Yuhong Liu, Zhan Liu, Yinquan Lu, Haijun Lv, Hongxia Lv, Huijie Lv, Qidang Lv, Ying Lv, Chengqi Lyu, Chenglong Ma, Jianpeng Ma, Ren Ma, Runmin Ma, Runyuan Ma, Xinzhu Ma, Yichuan Ma, Zihan Ma, Sixuan Mi, Junzhi Ning, Wenchang Ning, Xinle Pang, Jiahui Peng, Runyu Peng, Yu Qiao, Jiantao Qiu, Xiaoye Qu, Yuan Qu, Yuchen Ren, Fukai Shang, Wenqi Shao, Junhao Shen, Shuaike Shen, Chunfeng Song, Demin Song, Diping Song, Chenlin Su, Weijie Su, Weigao Sun, Yu Sun, Qian Tan, Cheng Tang, Huanze Tang, Kexian Tang, Shixiang Tang, Jian Tong, Aoran Wang, Bin Wang, Dong Wang, Lintao Wang, Rui Wang, Weiyun Wang, Wenhai Wang, Yi Wang, Ziyi Wang, Ling-I Wu, Wen Wu, Yue Wu, Zijian Wu, Linchen Xiao, Shuhao Xing, Chao Xu, Huihui Xu, Jun Xu, Ruiliang Xu, Wanghan Xu, GanLin Yang, Yuming Yang, Haochen Ye, Jin Ye, Shenglong Ye, Jia Yu, Jiashuo Yu, Jing Yu, Fei Yuan, Bo Zhang, Chao Zhang, Chen Zhang, Hongjie Zhang, Jin Zhang, Qiaosheng Zhang, Qiuyinzhe Zhang, Songyang Zhang, Taolin Zhang, Wenlong Zhang, Wenwei Zhang, Yechen Zhang, Ziyang Zhang, Haiteng Zhao, Qian Zhao, Xiangyu Zhao, Xiangyu Zhao, Bowen Zhou, Dongzhan Zhou, Peiheng Zhou, Yuhao Zhou, Yunhua Zhou, Dongsheng Zhu, Lin Zhu, and Yicheng Zou. Intern-s1: A scientific multimodal foundation model, 2025a. URL https://arxiv.org/abs/2508.15763.

Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, Wenbin Ge, Zhifang Guo, Qidong Huang, Jie Huang, Fei Huang, Binyuan Hui, Shutong Jiang, Zhaohai Li, Mingsheng Li, Mei Li, Kaixin Li, Zicheng Lin, Junyang Lin, Xuejing Liu, Jiawei Liu, Chenglong Liu, Yang Liu, Dayiheng Liu, Shixuan Liu, Dunjie Lu, Ruilin Luo, Chenxu Lv, Rui Men, Lingchen Meng, Xuancheng Ren, Xingzhang Ren, Sibo Song, Yuchong Sun, Jun Tang, Jianhong Tu, Jianqiang Wan, Peng Wang, Pengfei Wang, Qiuyue Wang, Yuxuan Wang, Tianbao Xie, Yiheng Xu, Haiyang Xu, Jin Xu, Zhibo Yang, Mingkun Yang, Jianxin Yang, An Yang, Bowen Yu, Fei Zhang, Hang Zhang, Xi Zhang, Bo Zheng, Humen Zhong, Jingren Zhou, Fan Zhou, Jing Zhou, Yuanzhi Zhu, and Ke Zhu. Qwen3-vl technical report, 2025b. URL https://arxiv.org/abs/2511.21631.

Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2.5-vl technical report. arXiv preprint arXiv:2502.13923, 2025c.

Mislav Balunović, Jasper Dekoninck, Ivo Petrov, Nikola Jovanović, and Martin Vechev. Matharena: Evaluating llms on uncontaminated math competitions, February 2025. URL https://matharena. ai/.

#### ByteDance-Seed. Beyondaime: Advancing math reasoning evaluation beyond high school olympiads. [https://huggingface.co/datasets/ByteDance-Seed/BeyondAIME](https: //huggingface.co/datasets/ByteDance-Seed/BeyondAIME), 2025.

Ding Chen, Qingchen Yu, Pengyuan Wang, Wentao Zhang, Bo Tang, Feiyu Xiong, Xinchi Li, Minchuan Yang, and Zhiyu Li. xverify: Efficient answer verifier for reasoning model evaluations. arXiv preprint arXiv:2504.10481, 2025a.

Jiacheng Chen, Qianjia Cheng, Fangchen Yu, Haiyuan Wan, Yuchen Zhang, Shenghe Zheng, Junchi Yao, Qingyang Zhang, Haonan He, Yun Luo, Yufeng Zhao, Futing Wang, Li Sheng, Chengxing Xie, Yuxin Zuo, Yizhuo Li, Wenxauan Zeng, Yulun Wu, Rui Huang, Dongzhan Zhou, Kai Chen, Yu Qiao, Lei Bai, Yu Cheng, Ning Ding, Bowen Zhou, Peng Ye, and Ganqu Cui. P1: Mastering physics olympiads with reinforcement learning, 2025b. URL https://arxiv.org/abs/2511.13612.

Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025.

Ganqu Cui, Lifan Yuan, Zefan Wang, Hanbin Wang, Wendi Li, Bingxiang He, Yuchen Fan, Tianyu Yu, Qixin Xu, Weize Chen, et al. Process reinforcement through implicit rewards. arXiv preprint arXiv:2502.01456, 2025a.

Ganqu Cui, Yuchen Zhang, Jiacheng Chen, Lifan Yuan, Zhi Wang, Yuxin Zuo, Haozhan Li, Yuchen Fan, Huayu Chen, Weize Chen, et al. The entropy mechanism of reinforcement learning for reasoning language models. arXiv preprint arXiv:2505.22617, 2025b.

#### Google Deepmind. Gemini-3-pro system card. URL https://deepmind.google/models/ gemini/pro/.

DeepSeek-AI, Aixin Liu, Aoxue Mei, Bangcai Lin, Bing Xue, Bing-Li Wang, Bingzheng Xu, Bochao Wu, Bowei Zhang, Chaofan Lin, Chen Dong, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenhao Xu, Chong Ruan, Damai Dai, Daya Guo, Dejian Yang, Deli Chen, Erhang Li, Fangqi Zhou, Fangyun Lin, Fucong Dai, Guangbo Hao, Guanting Chen, Guowei Li, H. Zhang, Hanwei Xu, Hao Li, Haofen Liang, Haoran Wei, Haowei Zhang, Hao sheng Luo, Haozhe Ji, Honghui Ding, Hongxuan Tang, Huan Cao, Huazuo Gao, Huixian Qu, Hui Zeng, Jialiang Huang, Jiashi Li, Jiaxin Xu, Jiewen Hu, JingChang Chen, Jingting Xiang, Jingyang Yuan, Jing Cheng, Jinhua Zhu, Jun Ran, Junguang Jiang, Junjie Qiu, Junlong Li, Jun-Mei Song, Kai Dong, Kaige Gao, Kang Guan, Kexin Huang, Kexing Zhou, Ke wei Huang, Kuai Yu, Lean Wang, Lecong Zhang, Lei Wang, Liang Zhao, Liangsheng Yin, Lihua Guo, Ling-Li Luo, Linwang Ma, Litong Wang, Liyue Zhang, M. S. Di, M. Y. Xu, Mingchuan Zhang, Minghua Zhang, Min Tang, Mingxu Zhou, P. Huang, Peixin Cong, Peiyi Wang, Qiancheng Wang, Qihao Zhu, Qingyang Li, Qinyu Chen, Qiushi Du, Ruiling Xu, Ruiqi Ge, Ruisong Zhang, Ruizhe Pan, Runji Wang, Runqiu Yin, Runxin Xu, Ruomeng Shen, Ruoyu Zhang, S. H. Liu, Shanghao Lu, Shangyan Zhou, Shanhuang Chen, Shaofei Cai, Shaoyuan Chen, Shengding Hu, Shengyu Liu, Shiqiang Hu, Shirong Ma, Shiyu Wang, Shuiping Yu, Shunfeng Zhou, Shuting Pan, Songyang Zhou, Tao Ni, Tao Yun, Tian Pei, Tian Ye, Tianyuan Yue, Wangding Zeng, Wen Liu, Wenfeng Liang, Wenjie Pang, Wenjing Luo, Wenjun Gao, Wentao Zhang, Xi Gao, Xiangwen Wang, Xiaoling Bi, Xiaodong Liu, Xiaohan Wang, Xiaokang Chen, Xiaokang Zhang, Xiaotao Nie, Xin Cheng, Xin Liu, Xin Xie, Xingchao Liu, Xingkai Yu, Xingyou Li, Xinyu Yang, Xinyuan Li, Xu Chen, Xuecheng Su, Xuehai Pan, Xuheng Lin, Xuwei Fu, Y. Q. Wang, Yang Zhang, Yanhong Xu, Yanru Ma, Yao Li, Yao Zhao, Yaofeng Sun, Yaohui Wang, Yi Qian, Yingpu Yu, Yichao Zhang, Yifan Ding, Yifan Shi, Yi Xiong, Ying He, Ying Zhou, Yinmin Zhong, Yishi Piao, Yisong Wang, Yixiao Chen, Yixuan Tan, Yixuan Wei, Yiyang Ma,

Yiyuan Liu, Yonglun Yang, Yongqiang Guo, Yongtong Wu, Yu Wu, Yuan Cheng, Yuan Ou, Yuanfan Xu, Yuduan Wang, Yue Gong, Yuhan Wu, Yu-Hui Zou, Yukun Li, Yunfan Xiong, Yu-Wei Luo, Yu mei You, Yuxuan Liu, Yuyang Zhou, Z. F. Wu, Zehui Ren, Zehua Zhao, Zehui Ren, Zhangli Sha, Zhe Fu, Zhean Xu, Zhenda Xie, Zhen guo Zhang, Zhewen Hao, Zhibin Gou, Zhicheng Ma, Zhigang Yan, Zhihong Shao, Zhixian Huang, Zhiyu Wu, Zhuoshu Li, Zhuping Zhang, Zian Xu, Zihao Wang, Zihui Gu, Zijia Zhu, Zi-Rui Li, Zipeng Zhang, Ziwei Xie, Ziyi Gao, Zizheng Pan, Zongqing Yao, Bei Feng, Hui Li, J. L. Cai, Jiaqi Ni, Lei Xu, Meng Li, Ning Tian, R. J. Chen, Ruiqi Jin, S. S. Li, Shuang Zhou, Tianyu Sun, X. Q. Li, Xiangyu Jin, Xiaojin Shen, Xiaosha Chen, Xinnan Song, Xinyi Zhou, Y. X. Zhu, Yanping Huang, Yao Li, Yi Zheng, Yuchen Zhu, Yunxiang Ma, Zhen Huang, Zhipeng Xu, Zhongyu Zhang, Dong-Li Ji, Jian Liang, Jianzhong Guo, Jin Chen, Leyi Xia, Miaojun Wang, Mingming Li, Peng Zhang, Ruyi Chen, Shangmian Sun, Shao-Kang Wu, Sheng-Ying Ye, T.Wang, W. L. Xiao, Wei An, Xianzu Wang, Xiaowen Sun, Xiaoxiang Wang, Ying Tang, Yukun Zha, Ze-Na Zhang, Zhenghua Ju, Zhen Zhang, and Zihua Qu. Deepseek-v3.2: Pushing the frontier of open large language models.

#### 2025. URL https://api.semanticscholar.org/CorpusID:283448719.

Shiyang Feng, Runmin Ma, Xiangchao Yan, Yue Fan, Yusong Hu, Songtao Huang, Shuaiyu Zhang, Zongsheng Cao, Tianshuo Peng, Jiakang Yuan, Zijie Guo, Zhijie Zhong, Shangheng Du, Weida Wang, Jinxin Shi, Yuhao Zhou, Xiaohan He, Zhiyin Yu, Fangchen Yu, Qihao Zheng, Jiamin Wu, Mianxin Liu, Chi Zhang, Shaowei Hou, Shuya Li, Yankai Jiang, Wenjie Lou, Lilong Wang, Zifu Wang, Jiong Wang, Wanghan Xu, Yue Deng, Dongrui Liu, Yiheng Wang, Wenlong Zhang, Fenghua Ling, Shufei Zhang, Xiaosong Wang, Shuangjia Zheng, Xun Huang, Siqi Sun, Shuyue Hu, Peng Ye, Chunfeng Song, Bin Wang, Conghui He, Yihao Liu, Xin Li, Qibin Hou, Tao Chen, Xiangyu Yue, Bin Wang, Liang He, Dahua Lin, Bowen Zhou, Bo Zhang, and Lei Bai. Internagent-1.5: A unified agentic framework for long-horizon autonomous scientific discovery. arXiv preprint arXiv:2602.08990,

###### 2026.

Elizabeth Gibney. Deepmind unveils ‘spectacular’general-purpose science ai. Nature, 641(8064): 827–828, 2025.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.

Yunzhuo Hao, Jiawei Gu, Huichen Will Wang, Linjie Li, Zhengyuan Yang, Lijuan Wang, and Yu Cheng. Can mllms reason in multimodality? emma: An enhanced multimodal reasoning benchmark, 2025. URL https://arxiv.org/abs/2501.05444.

Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with pagedattention. In Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles, 2023.

Hynek Kydlíček. Math-Verify: Math Verification Library. URL https://github.com/ huggingface/math-verify.

Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, et al. Deepseek-v3 technical report. arXiv preprint arXiv:2412.19437, 2024.

Jiacai Liu, Yingru Li, Yuqian Fu, Jiawei Wang, Qian Liu, and Yu Shen. When speed kills stability: Demystifying rl collapse from the training-inference mismatch. Notion Blog, 2025.

Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, KaiWei Chang, Michel Galley, and Jianfeng Gao. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. In International Conference on Learning Representations (ICLR), 2024.

Michael Luo, Sijun Tan, Justin Wong, Xiaoxiang Shi, William Y. Tang, Manan Roongta, Colin Cai, Jeffrey Luo, Li Erran Li, Raluca Ada Popa, and Ion Stoica. Deepscaler: Surpassing o1-preview with a 1.5b model by scaling rl. https://pretty-radio-b75.notion.site/ DeepScaleR-Surpassing-O1-Preview-with-a-1-5B-Model-by-Scaling-RL, 2025a. Notion Blog.

Yun Luo, Zhen Yang, Fandong Meng, Yafu Li, Jie Zhou, and Yue Zhang. An empirical study of catastrophic forgetting in large language models during continual fine-tuning. IEEE Transactions on Audio, Speech and Language Processing, 2025b.

Thang Luong, Dawsen Hwang, Hoang H. Nguyen, Golnaz Ghiasi, Yuri Chervonyi, Insuk Seo, Junsu Kim, Garrett Bingham, Jonathan Lee, Swaroop Mishra, Alex Zhai, Clara Huiyi Hu, Henryk Michalewski, Jimin Kim, Jeonghyun Ahn, Junhwi Bae, Xingyou Song, Trieu H. Trinh, Quoc V. Le, and Junehyuk Jung. Towards robust mathematical reasoning. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, 2025. URL https://aclanthology.org/2025.

#### emnlp-main.1794/.

Andrew Merry, Samyam Rajbhandari, Mohammad Shoeybi, Ramesh Puri, Paul Fung, Anima Anandkumar, and Bryan Catanzaro. Fully sharded data parallel: Reducing memory usage for large model training. PyTorch Developer Blog, 2021. URL https://pytorch.org/blog/ introducing-pytorch-fully-sharded-data-parallel-api/.

#### Meta. Llama 4 system card. URL https://www.llama.com/docs/ model-cards-and-prompt-formats/llama4/.

Aaron Meurer, Christopher P Smith, Mateusz Paprocki, Ondřej Čertík, Sergey B Kirpichev, Matthew Rocklin, AMiT Kumar, Sergiu Ivanov, Jason K Moore, Sartaj Singh, et al. Sympy: symbolic computing in python. PeerJ Computer Science, 3:e103, 2017.

Mistral. Mistral-medium-3 system card. URL https://mistral.ai/news/mistral-medium-3.

- OpenAI. Gpt-4o system card, a. URL https://openai.com/index/gpt-4o-system-card/.
- OpenAI. Gpt-5 system card, b. URL https://openai.com/index/gpt-5-system-card/. OpenAI. Gpt-5.2 system card, c. URL https://openai.com/index/introducing-gpt-5-2/.

OpenAI. Openai o3 and o4-mini system card, d. URL https://openai.com/index/ introducing-o3-and-o4-mini/.

Long Phan, Alice Gatti, Ziwen Han, Nathaniel Li, Josephina Hu, Hugh Zhang, Chen Bo Calvin Zhang, Mohamed Shaaban, John Ling, Sean Shi, et al. Humanity’s last exam. arXiv preprint arXiv:2501.14249, 2025.

Jiahao Qiu, Jingzhe Shi, Xinzhe Juan, Zelin Zhao, Jiayi Geng, Shilong Liu, Hongru Wang, Sanfeng Wu, and Mengdi Wang. Physics supernova: Ai agent matches elite gold medalists at ipho 2025. arXiv preprint arXiv:2509.01659, 2025.

David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, and Samuel R Bowman. Gpqa: A graduate-level google-proof q&a benchmark. In First Conference on Language Modeling, 2024.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu. Hybridflow: A flexible and efficient rlhf framework. arXiv preprint arXiv: 2409.19256, 2024.

Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick LeGresley, Jared Casper, and Bryan Catanzaro. Megatron-lm: Training multi-billion parameter language models using model parallelism. arXiv preprint arXiv:1909.08053, 2019.

Richard S Sutton, Andrew G Barto, et al. Reinforcement learning: An introduction, volume 1. MIT press Cambridge, 1998.

Richard S Sutton, David McAllester, Satinder Singh, and Yishay Mansour. Policy gradient methods for reinforcement learning with function approximation. Advances in neural information processing systems, 12, 1999.

GLM-V Team, Wenyi Hong, Wenmeng Yu, Xiaotao Gu, Guo Wang, Guobing Gan, Haomiao Tang, Jiale Cheng, Ji Qi, Junhui Ji, Lihang Pan, Shuaiqi Duan, Weihan Wang, Yan Wang, Yean Cheng, Zehai He, Zhe Su, Zhen Yang, Ziyang Pan, Aohan Zeng, Baoxu Wang, Bin Chen, Boyan Shi, Changyu Pang, Chenhui Zhang, Da Yin, Fan Yang, Guoqing Chen, Jiazheng Xu, Jiale Zhu, Jiali Chen, Jing Chen, Jinhao Chen, Jinghao Lin, Jinjiang Wang, Junjie Chen, Leqi Lei, Letian Gong, Leyi Pan, Mingdao Liu, Mingde Xu, Mingzhi Zhang, Qinkai Zheng, Sheng Yang, Shi Zhong, Shiyu Huang, Shuyuan Zhao, Siyan Xue, Shangqin Tu, Shengbiao Meng, Tianshu Zhang, Tianwei Luo, Tianxiang Hao, Tianyu Tong, Wenkai Li, Wei Jia, Xiao Liu, Xiaohan Zhang, Xin Lyu, Xinyue Fan, Xuancheng Huang, Yanling Wang, Yadong Xue, Yanfeng Wang, Yanzi Wang, Yifan An, Yifan Du, Yiming Shi, Yiheng Huang, Yilin Niu, Yuan Wang, Yuanchang Yue, Yuchen Li, Yutao Zhang, Yuting Wang, Yu Wang, Yuxuan Zhang, Zhao Xue, Zhenyu Hou, Zhengxiao Du, Zihan Wang, Peng Zhang, Debing Liu, Bin Xu, Juanzi Li, Minlie Huang, Yuxiao Dong, and Jie Tang. Glm-4.5v and glm-4.1vthinking: Towards versatile multimodal reasoning with scalable reinforcement learning, 2025a. URL https://arxiv.org/abs/2507.01006.

Kimi Team, Yifan Bai, Yiping Bao, Guanduo Chen, Jiahao Chen, Ningxin Chen, Ruijue Chen, Yanru Chen, Yuankun Chen, Yutian Chen, et al. Kimi k2: Open agentic intelligence. arXiv preprint arXiv:2507.20534, 2025b.

Miles Wang, Robi Lin, Kat Hu, Joy Jiao, Neil Chowdhury, Ethan Chang, and Tejal Patwardhan. Frontierscience: Evaluating ai’s ability to perform expert-level scientific tasks. arXiv preprint arXiv:2601.21165, 2026.

Weiyun Wang, Zhangwei Gao, Lixin Gu, Hengjun Pu, Long Cui, Xingguang Wei, Zhaoyang Liu, Linglin Jing, Shenglong Ye, Jie Shao, Zhaokai Wang, Zhe Chen, Hongjie Zhang, Ganlin Yang, Haomin Wang, Qi Wei, Jinhui Yin, Wenhao Li, Erfei Cui, Guanzhou Chen, Zichen Ding, Changyao Tian, Zhenyu Wu, Jingjing Xie, Zehao Li, Bowen Yang, Yuchen Duan, Xuehui Wang, Zhi Hou, Haoran Hao, Tianyi Zhang, Songze Li, Xiangyu Zhao, Haodong Duan, Nianchen Deng, Bin Fu, Yinan He, Yi Wang, Conghui He, Botian Shi, Junjun He, Yingtong Xiong, Han Lv, Lijun Wu, Wenqi Shao, Kaipeng Zhang, Huipeng Deng, Biqing Qi, Jiaye Ge, Qipeng Guo, Wenwei Zhang, Songyang Zhang,

Maosong Cao, Junyao Lin, Kexian Tang, Jianfei Gao, Haian Huang, Yuzhe Gu, Chengqi Lyu, Huanze Tang, Rui Wang, Haijun Lv, Wanli Ouyang, Limin Wang, Min Dou, Xizhou Zhu, Tong Lu, Dahua Lin, Jifeng Dai, Weijie Su, Bowen Zhou, Kai Chen, Yu Qiao, Wenhai Wang, and Gen Luo. Internvl3.5: Advancing open-source multimodal models in versatility, reasoning, and efficiency, 2025. URL https://arxiv.org/abs/2508.18265.

Xumeng Wen, Zihan Liu, Shun Zheng, Zhijian Xu, Shengyu Ye, Zhirong Wu, Xiao Liang, Yang Wang, Junjie Li, Ziming Miao, et al. Reinforcement learning with verifiable rewards implicitly incentivizes correct reasoning in base llms. arXiv preprint arXiv:2506.14245, 2025.

Colin White, Samuel Dooley, Manley Roberts, Arka Pal, Ben Feuer, Siddhartha Jain, Ravid Shwartz-Ziv, Neel Jain, Khalid Saifullah, Siddartha Naidu, et al. Livebench: A challenging, contamination-free llm benchmark. arXiv preprint arXiv:2406.19314, 4, 2024.

Zhiyu Wu, Xiaokang Chen, Zizheng Pan, Xingchao Liu, Wen Liu, Damai Dai, Huazuo Gao, Yiyang Ma, Chengyue Wu, Bingxuan Wang, et al. Deepseek-vl2: Mixture-of-experts vision-language models for advanced multimodal understanding. arXiv preprint arXiv:2412.10302, 2024.

xAI. Grok 4 system card. URL https://x.ai/grok. An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao,

Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.

Feng Yao, Liyuan Liu, Dinghuai Zhang, Chengyu Dong, Jingbo Shang, and Jianfeng Gao. Your efficient rl framework secretly brings you off-policy rl training, August 2025. URL https://fengyao. notion.site/off-policy-rl.

Fangchen Yu, Haiyuan Wan, Qianjia Cheng, Yuchen Zhang, Jiacheng Chen, Fujun Han, Yulun Wu, Junchi Yao, Ruilizhen Hu, Ning Ding, Yu Cheng, Tao Chen, Lei Bai, Dongzhan Zhou, Yun Luo, Ganqu Cui, and Peng Ye. Hipho: How far are (m)llms from humans in the latest high school physics olympiad benchmark? arXiv preprint arXiv:2509.07894, 2025a.

Fangchen Yu, Junchi Yao, Ziyi Wang, Haiyuan Wan, Youling Huang, Bo Zhang, Shuyue Hu, Dongzhan Zhou, Ning Ding, Ganqu Cui, Lei Bai, Wanli Ouyang, and Peng Ye. Physicsminions: Winning gold medals in the latest physics olympiads with a coevolutionary multimodal multi-agent system, 2025b. URL https://arxiv.org/abs/2509.24855.

Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, Lingjun Liu, et al. Dapo: An open-source llm reinforcement learning system at scale. arXiv preprint arXiv:2503.14476, 2025c.

Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, Cong Wei, Botao Yu, Ruibin Yuan, Renliang Sun, Ming Yin, Boyuan Zheng, Zhenzhu Yang, Yibo Liu, Wenhao Huang, Huan Sun, Yu Su, and Wenhu Chen. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. In Proceedings of CVPR, 2024a.

Xiang Yue, Tianyu Zheng, Yuansheng Ni, Yubo Wang, Kai Zhang, Shengbang Tong, Yuxuan Sun, Botao Yu, Ge Zhang, Huan Sun, Yu Su, Wenhu Chen, and Graham Neubig. Mmmu-pro: A more robust multi-discipline multimodal understanding benchmark. arXiv preprint arXiv:2409.02813, 2024b.

Qingyang Zhang, Haitao Wu, Changqing Zhang, Peilin Zhao, and Yatao Bian. Right question is already half the answer: Fully unsupervised llm reasoning incentivization. Advances in neural information processing systems, 2025.

Yu Zhang, Xiusi Chen, Bowen Jin, Sheng Wang, Shuiwang Ji, Wei Wang, and Jiawei Han. A comprehensive survey of scientific large language models and their applications in scientific discovery. arXiv preprint arXiv:2406.10833, 2024.

Chujie Zheng, Shixuan Liu, Mingze Li, Xiong-Hui Chen, Bowen Yu, Chang Gao, Kai Dang, Yuqiong Liu, Rui Men, An Yang, et al. Group sequence policy optimization. arXiv preprint arXiv:2507.18071, 2025a.

Lianmin Zheng, Liangsheng Yin, Zhiqiang Xie, Chuyue Livia Sun, Jeff Huang, Cody Hao Yu, Shiyi Cao, Christos Kozyrakis, Ion Stoica, Joseph E Gonzalez, et al. Sglang: Efficient execution of structured language model programs. Advances in neural information processing systems, 37:62557–62583, 2024.

Shenghe Zheng, Qianjia Cheng, Junchi Yao, Mengsong Wu, Haonan He, Ning Ding, Yu Cheng, Shuyue Hu, Lei Bai, Dongzhan Zhou, et al. Scaling physical reasoning with the physics dataset. arXiv preprint arXiv:2506.00022, 2025b.

Yizhen Zheng, Huan Yee Koh, Jiaxin Ju, Anh TN Nguyen, Lauren T May, Geoffrey I Webb, and Shirui Pan. Large language models for scientific discovery in molecular property prediction. Nature Machine Intelligence, pages 1–11, 2025c.

Jinguo Zhu, Weiyun Wang, Zhe Chen, Zhaoyang Liu, Shenglong Ye, Lixin Gu, Hao Tian, Yuchen Duan, Weijie Su, Jie Shao, et al. Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models. arXiv preprint arXiv:2504.10479, 2025a.

Zilin Zhu, Chengxing Xie, Xin Lv, and slime Contributors. slime: An llm post-training framework for rl scaling. https://github.com/THUDM/slime, 2025b. GitHub repository. Corresponding author: Xin Lv.

### A. Appendix

###### A.1. RL Training on Intern Series

We further verify the effectiveness of our RL training on the InternVL-3.5-30B-A3B and Intern-S1-mini models. On the HiPhO benchmark, InternVL-3.5-30B-A3B performance improves from 18.3 to 22.6, while Intern-S1-mini sees an increase from 8.5 to 12.4 (utilizing Qwen3-30B-A3B-Instruct as the model-based verifier). Using Gemini-2.5-Flash as the verifier, the trained Intern-S1-mini achieves one silver and ten bronze medals, an improvement over the baseline of ten bronze medals. These consistent improvements demonstrate that our RL training strategy is effective in eliciting the models’ latent capabilities in scientific reasoning.

###### A.2. Case Study I: Text+Variable Figure

This case study examines a complex problem from the 2025 International Physics Olympiad (IPhO), which investigates the physical principles of Cox’s 18th-century timepiece—an ingenious device that harnesses atmospheric pressure fluctuations to generate energy. The problem requires understanding the figure information to determine the states of the tubes and the behaviors, and analyzing the table information. The P1-VL-235B-A22B model achieved a perfect score (1.0/1.0) on this problem, demonstrating strong capabilities in:

- • Physical intuition: Correctly identifying the critical force balance constraint at the stop position.
- • Visual-Structural Alignment: Accurately translating the schematic representation of the tube states (liquid levels and gas volumes) into precise hydrostatic equations, effectively bridging visual features with symbolic reasoning.
- • Tabular Information Integration: Synthesizing discrete parameters from the provided table to constrain the variables, specifically extracting the dimensional specifications required to calculate the total work done.
- • Long-horizon Logical Reasoning: Sustaining a coherent derivation chain to link atmospheric pressure fluctuations with mechanical energy storage, correctly applying the Ideal Gas Law without hallucinating non-existent constraints.

This performance demonstrates P1-VL’s proficiency in tackling competition-level physics problems, requiring a synergy of deep physical understanding, rigorous analytical reasoning, and precise diagrammatic interpretation.

- IPhO 2025 Question 2-A.1

Background:

In 1765, British clockmaker James Cox invented a clock whose only source of energy is the fluctuations in atmospheric pressure. Cox’s clock used two vessels containing mercury. Changes in atmospheric pressure caused mercury to move between the vessels, and the two vessels to move relative to each other. This movement acted as an energy source for the actual clock.

We propose an analysis of this device. Throughout, we assume that

- • the Earth’s gravitational field 𝑔 = −𝑔 𝑢𝑧 is uniform with 𝑔 = 9.8m · s−2 and 𝑢𝑧 a unit vector;
- • all liquids are incompressible and their density is denoted 𝑟ℎ𝑜;
- • no surface tension effects will be considered; - the variations of atmospheric pressure with altitude are neglected;

- • the surrounding temperature 𝑇a is uniform and all transformations are isothermal.

We first consider a bath of water that occupies the semi-infinite space 𝑧 ≤ 0. The air above it is at a pressure 𝑃a = 𝑃0. A cylindrical vertical tube of length 𝐻 = 1m, cross-sectional area 𝑆 = 10cm2 and mass 𝑚 = 0.5kg is dipped into the bath. The bottom end of the tube is open, and the top end of the tube is closed. We denote ℎ the altitude of the top of the tube and 𝑧ℓ that of the water inside the tube. The thickness of the tube walls is neglected.

We start from the situation where the tube in Figure 11 contains no gas and its top is at the bath level: in other words, ℎ = 0 and 𝑧𝑙 = 0 (case a). The tube is then slowly lifted until its bottom end reaches the bath level. The pulling force exerted on the tube is denoted 𝐹 = 𝐹 𝑢𝑧.

[Figure 6]

- Figure 11 | Sketch of the tube in different configurations

##### Question:

For the configuration shown in Figure 11 (case b), determine the expressions for:

- 1. The pressure 𝑃𝑤 in the water at the top of the tube
- 2. The force 𝐹 necessary to maintain the tube at this position

Given data: 𝑃0, 𝜌, 𝑚, 𝑆, ℎ, 𝑔 and 𝑢𝑧

###### P1-VL Solution Overview

To analyze the situation for case b in Cox’s timepiece, we consider the vertical tube of length 𝐻, cross-sectional area 𝑆, and mass 𝑚, lifted so that its bottom end is at the bath surface (𝑧 = 0) and its top is at altitude ℎ. In this configuration, the water level inside the tube is at the same height as the top of the tube, i.e., 𝑧ℓ = ℎ, indicating that the tube is completely filled with water.

###### (A.1) 𝑃w and 𝐹

The tube is sealed at the top and open at the bottom, and the water inside is in hydrostatic equilibrium. The pressure at the bottom of the tube (at 𝑧 = 0) must equal the atmospheric pressure 𝑃0, as the tube is open to the bath at this level.

Using the hydrostatic pressure law, the pressure at any height 𝑧 inside the water column is given by:

𝑃(𝑧) = 𝑃0 − 𝜌𝑔𝑧 (11) At the top of the tube, where 𝑧 = ℎ, the pressure in the water is:

𝑃w = 𝑃0 − 𝜌𝑔ℎ (12)

This result reflects that the pressure at the top of the water column is reduced by the weight of the water column of height ℎ.

To determine the force 𝐹 necessary to hold the tube in place, we perform a force balance on the tube alone. Forces on the Tube:

- • Gravitational force: The tube has a mass 𝑚, so its weight is −𝑚𝑔 𝑢𝑧.
- • Pressure forces: The atmospheric pressure 𝑃0 acts downward on the top cap of the tube, exerting a force −𝑃0𝑆 𝑢𝑧.The water pressure 𝑃w at the top of the tube acts upward on the inner surface of the cap, exerting a force +𝑃w𝑆 𝑢𝑧.
- • Applied force: 𝐹 is applied upward to maintain equilibrium.

Thus, the net pressure force on the tube is:

(𝑃w − 𝑃0)𝑆 𝑢𝑧 = −𝜌𝑔ℎ𝑆 𝑢𝑧 (13)

This is a downward force due to the pressure difference between the outside atmosphere and the water column inside the tube. The net force on the tube must be zero:

𝐹 + (𝑃w − 𝑃0)𝑆 𝑢𝑧 − 𝑚𝑔 𝑢𝑧 = 0 (14) Substituting 𝑃w = 𝑃0 − 𝜌𝑔ℎ:

𝐹 − 𝜌𝑔ℎ𝑆 𝑢𝑧 − 𝑚𝑔 𝑢𝑧 = 0 ⇒ 𝐹 = (𝑚𝑔 + 𝜌𝑔ℎ𝑆) 𝑢𝑧 (15)

This result is physically intuitive: the force required to hold the tube in place must counteract both the tube’s own weight and the net downward pressure force due to the difference between atmospheric pressure and the reduced pressure at the top of the water column.

|𝐹 = (𝑚𝑔 + 𝜌𝑔ℎ𝑆) 𝑢𝑧|
|---|

###### Answer:

|𝑃w = 𝑃0 − 𝜌𝑔ℎ|
|---|

##### IPhO 2025 Question 2-A.2

Background: Three experiments are performed. In each, the tube is lifted from the initial state shown in Figure 11(a) under the conditions specified in Figure 12.

[Figure 7]

- Figure 12 | Experimental conditions and numerical values of physical quantities for each experiment (𝑃sat designates the saturated vapour pressure of the pure fluid)

In each case, we study the evolution of the force 𝐹 that must be applied in order to maintain the tube in equilibrium at an altitude ℎ, the external pressure being fixed at 𝑃a = 𝑃0 = 1.000×105 Pa. Two different behaviours are possible

[Figure 8]

- Figure 13 | Two different behaviours

##### Question:

For each experiment, complete the table in the answer sheet to indicate the expected behaviour and the numerical values for 𝐹max and for ℎ∗ (when pertinent), where 𝐹max and ℎ∗ are defined in the figures illustrating the two behaviours.

###### P1-VL Solution Overview

To determine the behavior of the tube during lifting and the corresponding values of 𝐹max and ℎ∗, we analyze the pressure equilibrium and the conditions under which vaporization (cavitation) occurs.

###### (A.2) Answer Sheet

The vertical cylinder, sealed at the top and open at the bottom, is lifted from the bath (where ℎ = 0) to the position where its bottom end reaches the surface (i.e., ℎ = 𝐻 = 1m). The pressure at the top of the tube is governed by:

𝑃top = 𝑃0 − 𝜌𝑔ℎ (16) Cavitation (formation of vapor) occurs when 𝑃top = 𝑃sat, i.e., when:

𝑃0 − 𝑃sat 𝜌𝑔

(17)

ℎ∗ =

- • Behavior A occurs when ℎ∗ > 𝐻, meaning the tube remains completely filled with liquid throughout the lifting process (𝑧ℓ = ℎ).
- • Behavior B occurs when ℎ∗ < 𝐻, meaning the water column inside the tube stops rising at ℎ∗, and the top of the tube continues to rise while the water level inside remains at 𝑧ℓ = ℎ∗.

The force 𝐹 required to hold the tube in equilibrium is given by: 𝐹 = 𝑚𝑔 + (𝑃0 − 𝑃top)𝑆 (18)

- • For Behavior A, 𝑃top = 𝑃0 − 𝜌𝑔ℎ, so: 𝐹 = 𝑚𝑔 + 𝜌𝑔𝑆ℎ (19)

At ℎ = 𝐻,

𝐹max = 𝑚𝑔 + 𝜌𝑔𝑆𝐻 (20)

- • For Behavior B, once ℎ > ℎ∗, 𝑃top = 𝑃sat, so: 𝐹max = 𝑚𝑔 + (𝑃0 − 𝑃sat)𝑆 (21)

###### Data Recap

- • Cross-sectional area: 𝑆 = 10cm2 = 1 × 10−3 m2
- • Mass of the tube: 𝑚 = 0.5kg
- • Gravitational acceleration: 𝑔 = 9.8m/s2
- • Atmospheric pressure: 𝑃0 = 1.000 × 105 Pa

###### Experiment 1: Water at 20◦C

- • 𝜌 = 1.00 × 103 kg/m3
- • 𝑃sat = 2.34 × 103 Pa

1.000 × 105 − 2.34 × 103 1.00 × 103 × 9.8 ≈ 9.96m > 𝐻 (22)

ℎ∗ =

→ Behavior A

𝐹max = 𝑚𝑔 + 𝜌𝑔𝑆𝐻 = 0.5 × 9.8 + 1.00 × 103 × 9.8 × 1 × 10−3 = 4.9 + 9.8 = 14.7N (23)

###### Experiment 2: Water at 80◦C

- • 𝜌 = 0.97 × 103 kg/m3
- • 𝑃sat = 47.4 × 103 Pa ℎ∗ =

1.000 × 105 − 47.4 × 103 0.97 × 103 × 9.8 ≈ 5.53m > 𝐻 (24)

- → Behavior A 𝐹max = 𝑚𝑔 + 𝜌𝑔𝑆𝐻 = 4.9 + 0.97 × 103 × 9.8 × 1 × 10−3 ≈ 4.9 + 9.506 = 14.4N (25)

Experiment 3: Water at 99◦C

- • 𝜌 = 0.96 × 103 kg/m3
- • 𝑃sat = 99.8 × 103 Pa

ℎ∗ =

1.000 × 105 − 99.8 × 103 0.96 × 103 × 9.8 ≈ 0.0213m = 2.1cm < 𝐻 (26)

- → Behavior B 𝐹max = 𝑚𝑔 + (𝑃0 − 𝑃sat)𝑆 = 4.9 + (200)(1 × 10−3) = 4.9 + 0.2 = 5.1N (27)

###### Answer:

|Experiment<br><br>|Behaviour (A or B?)|ℎ∗ (cm)<br><br>|𝐹max (N)|
|---|---|---|---|
|1<br><br>2<br><br>3<br><br><br>|A<br><br>A<br><br>B<br><br><br>|None None 2.1|14.7 14.4 5.1<br><br>|

