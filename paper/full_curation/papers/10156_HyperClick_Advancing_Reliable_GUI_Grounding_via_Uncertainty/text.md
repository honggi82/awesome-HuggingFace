# arXiv:2510.27266v2[cs.CV]27May2026

## Enhancing Trustworthy GUI Grounding via Self-Critiqued Reinforcement Learning

### Shaojie Zhang∗, Pei Fu∗, Ruoceng Zhang, Jiahui Yang, Anan Du, Xiuwen Xi, Shaokang Wang, Ying Huang, Bin Qin, Zhenbo Luo†, Jian Luan

MiLM Plus, Xiaomi Inc {zhangshaojie5, fupei1, luozhenbo, luanjian}@xiaomi.com

### Abstract

Autonomous graphical user interface (GUI) agents rely on accurate GUI grounding, which maps language instructions to on-screen coordinates, to execute user commands. However, current models, whether trained via supervised fine-tuning (SFT) or reinforcement learning (RL), often provide confidence signals that are poorly aligned with actual grounding correctness, leading to overconfident and unreliable predictions. To address this, we propose HyperClick, a novel framework that enhances trustworthy GUI grounding through self-critiqued reinforcement learning (SCRL). HyperClick combines a correctness reward and a confidence alignment reward, training the policy model to output both a click prediction and an explicit confidence estimate. This approach jointly optimizes grounding accuracy and confidence reliability through confidencebased self-assessment. Extensive experiments on challenging benchmarks show that HyperClick maintains strong grounding performance while providing better-aligned confidence estimates. By exposing uncertainty alongside GUI actions, HyperClick supports confidence-based abstention in GUI automation. Code will be released here.

### 1 Introduction

The revolution of autonomous graphical user interface (GUI) agents is transforming human-computer interaction, allowing users to control mobile applications, web platforms, and complex desktop software directly through natural language instructions (Wang et al., 2024b; Nguyen et al., 2024). At the heart of these agents lies the GUI grounding, the ability to accurately map textual commands to precise pixel coordinates on user interface elements (Cheng et al., 2024; Tang et al., 2025). This fundamental task determines whether an agent can successfully execute user commands, making it the cornerstone of the GUI automation.

Recent progress in GUI grounding has been driven by supervised fine-tuning (SFT) with curated large-scale datasets (Wu et al., 2024; Gou et al., 2025; Xu et al., 2024) and reinforcement learning (RL) with verifiable GUI-specific rewards (Lu et al., 2025; Luo et al., 2025; Liu et al., 2025b). Although these techniques yield strong performance, their confidence estimates are often poorly aligned with actual grounding correctness, making it difficult to judge when predictions are reliable.

A trustworthy GUI agent should be aware of its limitations and distinguish accurately between what it can and cannot do (Ding et al., 2025). Although confidence estimation has been extensively studied in large language models (LLMs) (Xiong et al., 2023; Tian et al., 2023), it remains underexplored in GUI agents. The reliability level of an agent can be assessed by the alignment between its confidence and actual performance (Ding et al., 2025). In this paper, we first evaluate probabilistic and verbalized confidence for several general models (OpenAI, 2024; Bai et al., 2025; Guo et al., 2025b; Team et al., 2025; Xiaomi, 2025) and GUI-specific models (Qin et al., 2025) on the ScreenSpot-Pro benchmark (Li et al., 2025), which emphasizes high-resolution displays, smaller target sizes, and complex environments. Specifically, probabilistic confidence reflects token-level likelihoods for predicted coordinates (Guo et al., 2017; Desai and Durrett, 2020), while verbalized confidence captures self-reported certainty in natural language (Lin et al., 2022; Yang et al., 2024b).

As shown in Figure 1, the models exhibit a higher confidence in their answers than in the accuracy that they actually achieve. In other words, even on challenging tasks, these agents remain overconfident in their predictions, both probabilistically and in their self-assessments. This resembles the broader reliability challenge commonly observed in LLMs and vision-language models (VLMs), where

[Figure 1]

Probabilistic Confidence

[Figure 2]

0.748 0.463

······

- Round 1

- Round 2

[Figure 3]

User:Click Privacy & security.

Assistant:[350,2100]

Verbalized Confidence

response element

User: Output a float number ranging from 0. to 1. representing your confidence with your provided answer.

| |
|---|

[Figure 4]

Assistant:0.816

screenshot

(a) (b)

- Figure 1: Overview of accuracy and confidence evaluation on ScreenSpot-Pro. (a): Illustration of probabilistic and verbalized confidence. Probabilistic confidence represents the probability of the model generating the next token corresponding to the target coordinates, while verbalized confidence indicates the model’s self-reported certainty about its output in natural language. (b): Comparisons of accuracy, probabilistic confidence, and verbalized confidence for several general-purpose and GUI-specific models on the ScreenSpot-Pro benchmark. The models exhibit a higher confidence in their answers than in the accuracy that they actually achieve.

models can produce erroneous outputs while maintaining high confidence (Ji et al., 2023a,b; Kalai et al., 2025). This limitation is particularly critical in real-world GUI tasks, where their dynamic, continuous nature means that even a single error at an intermediate step can result in overall task failure.

To address this limitation, we propose HyperClick, a novel framework that enhances trustworthy GUI grounding through self-critiqued reinforcement learning (SCRL). Unlike prior approaches that treat grounding as a binary, hit-or-miss classification task, HyperClick explicitly integrates confidence estimation into the decision-making process. Specifically, each prediction delivers not only a targeted UI element but also a verbalized confidence statement that serves as a reliable self-assessment. In this work, we deliberately focus on aligning verbalized confidence rather than probabilistic confidence. While the latter relies heavily on token likelihoods and is easily confounded by coordinate tokenization or numeric formatting, verbalized confidence is entirely model-agnostic and can be directly exposed to downstream GUI agents, making it a cleaner reflection of spatial click quality.

Specifically, we introduce two complementary rule-based reward mechanisms that optimize both action precision and confidence alignment. A binary reward enforces correct grounding positions, while a confidence alignment reward aligns the verbalized confidence of policy models with a bounded spatial confidence target constructed over the annotated element region. This dual mechanism enables HyperClick to achieve two inter-

twined goals: accurate GUI grounding and betteraligned confidence. By training the model to expose uncertainty alongside its click prediction, HyperClick reduces high-confidence errors and enables confidence-based abstention for safer GUI decision-making.

Our contributions are summarized as follows:

- • We systematically reveal that existing GUI grounding models are prone to overconfident predictions and highlight their implications for reliable GUI automation.
- • We propose HyperClick, a trustworthy GUI grounding framework that integrates SCRL, introducing a dual reward mechanism that jointly optimizes grounding accuracy and confidence alignment via binary correctness and a bounded spatial confidence target.
- • Through extensive evaluations on challenging GUI grounding benchmarks, HyperClick maintains strong accuracy while producing confidence-quality-aligned estimates that enable selective abstention on uncertain clicks.

### 2 Related Work

#### 2.1 GUI Agents and GUI Grounding

GUI agents automate desktop and mobile tasks by interacting with graphical user interfaces through natural language instructions (Wang et al., 2024b; Nguyen et al., 2024; Zhang et al., 2024). Recent VLM-based agents (Cheng et al., 2024; Wu et al., 2024; Qin et al., 2025) combine visual perception

with language reasoning to handle diverse interface styles. Their reliability largely depends on GUI grounding, which maps instructions to precise interface elements or pixel coordinates.

Early studies mainly acquire GUI-specific abilities through supervised fine-tuning on large-scale GUI corpora (Cheng et al., 2024; Lin et al., 2025;

- Yang et al., 2024a). SeeClick (Cheng et al., 2024) introduces visual-only GUI grounding, while OSAtlas (Wu et al., 2024), UGround (Gou et al., 2025), and Aguvis (Xu et al., 2024) improve perception by fine-tuning pretrained models on diverse GUI environments. UI-TARS (Qin et al., 2025) further scales this direction toward an end-to-end GUI agent for unified cross-platform action modeling.

Recent RL work further extends GUI grounding with verifiable rewards inspired by reasoning models (Guo et al., 2025a). R1-style GUI agents optimize policies with such rewards (Lu et al., 2025; Luo et al., 2025; Liu et al., 2025b; Zhang et al., 2025b), often requiring explicit reasoning before prediction. Later methods tailor reward design to spatial grounding through controllable box-size rewards (Zhou et al., 2025), self-evolution with continuous rewards (Yuan et al., 2025), and Gaussian reward modeling (Tang et al., 2025). These methods improve grounding accuracy, but they primarily optimize localization itself and largely overlook whether model confidence is aligned with grounding quality.

- 2.2 Confidence in LLMs

Confidence quantifies prediction reliability and has been widely used in error analysis (Oberkampf et al., 2002) and computer vision tasks such as object detection and segmentation (Ren et al., 2015; Redmon et al., 2016; Long et al., 2015; He et al., 2017). For LLMs and VLMs, existing confidence signals can be broadly grouped into probabilistic confidence from token probabilities (Guo et al., 2017; Desai and Durrett, 2020), answer consistency confidence from agreement among multiple outputs (Zhang et al., 2023; Manakul et al., 2023; Fu et al., 2025), and verbalized confidence from explicit self-reported certainty (Lin et al., 2022;

- Yang et al., 2024b). Probabilistic and consistency-based confidence

often correlate better with model performance, but they require token likelihoods, repeated sampling, or in-domain calibration. Verbalized confidence is easier to expose to downstream agents and is model-agnostic, yet it is prone to weak alignment

and overconfidence. Therefore, we align verbalized confidence with a spatial confidence distribution over the screenshot, turning self-critiqued confidence into a grounding-quality signal for selective abstention.

### 3 Method

#### 3.1 Problem Formulation

GUI grounding maps a natural language instruction to the spatial coordinates of the target UI element. From a policy optimization view, it is commonly instantiated as either location formulation (Wu et al., 2024; Tang et al., 2025) or click formulation (Xu et al., 2024; Luo et al., 2025; Yuan et al., 2025).

- • Location formulation: Given a screenshot s and an instruction q, the policy model is optimized by predicting the bounding box bˆ =

(ˆx1,yˆ1,xˆ2,yˆ2), where (ˆx1,yˆ1) and (ˆx2,yˆ2) denote the top-left and bottom-right corners of the UI element referred to by q.

- • Click formulation: Alternatively, the policy model predicts a single point pˆ = (ˆx,yˆ), corresponding to the center of the target element, which directly simulates a clicking action.

We adopt the click formulation because it directly matches executable GUI actions, reduces the action space, and provides a natural reinforcement learning objective.

#### 3.2 Confidence Modeling

Following Gaussian error analysis (Gauss, 1809, 1877; MacKenzie, 1988), we construct a bounded spatial confidence representation for GUI grounding using a Gaussian kernel over the annotated element region. This target is not intended to be an absolute probability; instead, it provides a smooth supervision signal for confidence alignment, assigning higher values to clicks near the annotated element center and lower values to less precise clicks. We refer to this objective as confidencequality alignment: confidence that ranks predictions by grounding success and supports selective abstention, rather than an absolute probability over arbitrary failure modes. Since most GUI annotations are bounding boxes, this target naturally combines two requirements: a click outside the target box should receive zero confidence, while clicks inside the box can still be differentiated by their spatial proximity to the center.

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

Completions Generation Verifiable Reward

Advantage Estimation

Instruction: Click Privacy & security.

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

<point>[350,2100]</point> <confidence>0.969</confidence>

······ ······

Policy Model

[Figure 23]

[Figure 24]

response element

[Figure 25]

[Figure 26]

[Figure 27]

Correctness Reward Confidence Alignment Reward

| |
|---|

[Figure 28]

[Figure 29]

response element

| |
|---|

[Figure 30]

[Figure 31]

Spatial Confidence Representation

Binary Representation

- Figure 2: Framework of the proposed HyperClick, optimized with Group Relative Policy Optimization (GRPO). Given a screenshot and an instruction, the policy generates N predictions, which are evaluated by a verifiable reward mechanism. The correctness reward measures grounding precision, while the confidence alignment reward aligns the verbalized confidence of the policy model with a bounded spatial confidence target over the annotated element region. For clarity, the reference model is omitted.

For a target box b, let µ = (µx,µy) denote its center and let σx,σy control a diagonal covariance. For each predicted point pˆ = (ˆx,yˆ), the base spatial confidence target is:

2

2

σx2 + (ˆy−µy)

C(pˆ) = exp − 12 (ˆx−µx)

σy2 . (1)

The value is maximized at the ground-truth center. We then constrain the target to the bounding box b by setting C(pˆ) = C(pˆ) when pˆ ∈ b and C(pˆ) = 0 otherwise. This preserves the binary correctness boundary used by GUI grounding benchmarks while providing a graded target for confidence learning inside valid regions. Adaptive Variance. Prior work (Zhou et al., 2025; Tang et al., 2025) shows that smaller UI elements are harder to ground. To handle diverse element sizes, we scale the variance by the box dimensions:

σx = α · (x2 − x1), σy = α · (y2 − y1), (2)

where α controls how element size affects the standard deviations. As a result, larger elements receive broader high-confidence regions, while smaller elements produce sharper confidence peaks that reflect the higher precision required for successful clicks.

#### 3.3 Training Objective

Format Reward. We constrain the policy to emit each response in the format <point>[x,y] </point><confidence>conf</confidence>,

where conf is rounded to three decimal places.

Rformat returns 1 when the completion matches this template and 0 otherwise, preventing overly long completions and enabling unambiguous parsing.

Correctness Reward. We use a binary reward to guide the predicted point pˆ into the bounding box b, directly matching the success criterion of GUI grounding:

 

1, if (x1 < xˆ < x2)

∧ (y1 < yˆ < y2), 0, otherwise.

Rcorrectness = pˆ∈b =



(3)

Confidence Alignment Reward. The confidence alignment reward encourages the policy to evaluate its own prediction. We align the generated confidence cˆ with the spatial confidence target in Section 3.2 using an l2 constraint:

Rconfidence = 1 − (ˆc − C(ˆx,yˆ))2. (4)

The reward is bounded because both cˆand C(·) lie in [0,1], and it favors confidence values close to the constructed target. It also rewards low confidence on incorrect predictions, encouraging confidencequality alignment rather than overconfident clicks.

The final reward combines format, correctness, and confidence alignment terms:

R = Rformat + Rcorrectness + Rconfidence. (5)

We optimize HyperClick with Group Relative Policy Optimization (GRPO) (Shao et al., 2024).

Given N generations {oi}Ni=1, GRPO evaluates each output with R and normalizes rewards within the group to obtain relative advantages:

R(oi) − mean({R(oj)}Nj=1) std({R(oj)}Nj=1)

(6)

Ai =

The training objective of GRPO is then defined as

J (θ) = E{oi}∼πθ

old(·|s,q)

N

1 N

min ri(θ)Ai,

i=1

clip(ri(θ),1 − ϵ,1 + ϵ)Ai

− β KL(πθ ∥πref),

(7)

where πθ is the policy model, ϵ controls clipping, and β weights the KL regularization (Schulman et al., 2017; Shao et al., 2024).

### 4 Experiments

#### 4.1 Implementation Details

We implement HyperClick based on Qwen2.5-VL3B-Instruct and Qwen2.5-VL-7B-Instruct. Model training is conducted within the VLM-R1 (Shen et al., 2025) codebase. We train for one epoch on 16 NVIDIA H100 GPUs, using a cosine schedule decaying from 1e-6 to 0, a global batch size of 16, 8 generations per instance, and a KL constraint coefficient of β = 0.04. To improve efficiency, we leverage FlashAttention-2 (Dao, 2023), adopt bfloat16 precision, and enable gradient checkpointing. During inference, the temperature is fixed to 0 to ensure reproducibility. Full details of the training data, including source datasets and the RL sample construction procedure, are provided in the Appendix.

#### 4.2 Evaluation Benchmarks

We comprehensively evaluated HyperClick’s GUI grounding capability across four benchmarks. ScreenSpot (Cheng et al., 2024) evaluates the grounding of the GUI on mobile, desktop, and web platforms, providing a diverse set of interface types for comparing robustness across common user scenarios. ScreenSpot-V2 (Wu et al., 2024) extends ScreenSpot with more challenging tasks and refined annotations, and tests grounding accuracy in various real-world environments. ScreenSpotPro (Li et al., 2025) focuses on high-resolution professional settings with expert-annotated tasks,

covering 23 applications, five industries, and three operating systems. MMBench-GUI (Wang et al., 2025) organizes tasks into a hierarchical structure of basic and advanced instructions, enabling systematic evaluation across instruction complexity levels.

#### 4.3 Main Results

Comparisons with recent baselines. We present the main results of our evaluation in Table 1, 2, 3. The results show that HyperClick maintains competitive performance among open-source models in both 3B and 7B parameter categories while adding explicit confidence-quality alignment. In particular, HyperClick demonstrates consistent gains across diverse platforms and task settings, including mobile, desktop, and web environments. The benefits are especially visible in challenging benchmarks, such as ScreenSpot-Pro and MMBenchGUI, which feature high-resolution interfaces, small target elements, and complex visual layouts. This suggests that HyperClick is effective in scenarios where precise spatial grounding and robustness are critical.

A key source of HyperClick’s improvement lies in the introduction of SCRL, which trains the model to pair each click prediction with an explicit confidence estimate. Unlike prior RL-based GUI grounding models that rely solely on sparse binary (Lu et al., 2025; Luo et al., 2025) or continuous (Yuan et al., 2025; Tang et al., 2025) correctness rewards, HyperClick leverages confidence alignment to distinguish reliable clicks from uncertain ones. This enables policy models to penalize overconfident errors while reinforcing well-aligned predictions. The resulting training objective yields consistent gains across benchmarks, suggesting that confidence alignment can improve generalization across diverse UI environments. These results show that trustworthy GUI grounding can enhance accuracy while providing a spatially grounded ranking signal that correlates with grounding success.

HyperClick improves confidence alignment. Figure 3 reports reliability diagrams on ScreenSpotPro. For each confidence bin, the blue bar denotes empirical grounding accuracy, the red hatched region denotes the gap to the perfect-alignment diagonal, and the expected Calibration Error (ECE, ↓) (Naeini et al., 2015; Guo et al., 2017) summarizes the sample-weighted alignment gap across bins. The baseline models are severely overconfident, with large gaps in high-confidence bins and

- Table 1: GUI grounding accuracy on the ScreenSpot (Cheng et al., 2024) and ScreenSpot-V2 (Cheng et al., 2024) benchmarks over the Mobile, Desktop, and Web sub-tasks. Bold and underline indicate the best and second-best results.

Model Size

ScreenSpot

SS Avg.

ScreenSpot V2

SSv2 Mobile Desktop Web Mobile Desktop Web Avg.

Text Icon Text Icon Text Icon Text Icon Text Icon Text Icon General Models

GPT-4o (OpenAI, 2024) - 30.5 23.2 20.6 19.4 11.1 7.8 18.8 26.6 24.2 24.2 19.3 12.8 11.8 20.1 Qwen2.5-VL (Bai et al., 2025) 7B - - - - - - 84.7 97.6 87.2 90.2 74.2 93.2 81.3 88.8

GUI-specific Models (SFT)

SeeClick (Cheng et al., 2024) 9.6B 78.0 52.0 72.2 30.0 55.7 32.5 53.4 78.4 50.7 70.1 29.3 55.2 32.5 55.1 UGround (Gou et al., 2025) 7B 82.8 60.3 82.5 63.6 80.4 70.4 73.3 75.1 84.5 85.1 61.4 84.6 71.9 76.3 OS-Atlas (Wu et al., 2024) 7B 93.0 72.9 91.8 62.9 90.89 74.3 82.5 95.2 75.8 90.7 63.6 90.6 77.3 84.1 UI-TARS (Qin et al., 2025) 7B 94.5 85.2 95.9 85.7 90.0 83.5 89.5 96.9 89.1 95.4 85.0 93.6 85.2 91.6 TongUI (Zhang et al., 2025a) 7B 91.9 79.5 93.8 80.0 89.1 81.6 86.0 93.1 81.5 96.4 82.9 90.2 84.7 88.7 GUI-Actor (Wu et al., 2025) 7B 94.9 82.1 91.8 80.0 91.3 85.4 88.3 96.5 84.3 91.7 84.1 93.9 82.3 89.5

GUI-specific Models (RL)

UI-R1 (Lu et al., 2025) 3B 95.6 84.7 90.2 59.3 85.2 73.3 83.3 96.2 84.3 92.3 63.6 89.2 75.4 85.4 UI-R1-E (Lu et al., 2025) 3B 97.1 83.0 95.4 77.9 91.7 85.0 89.2 98.2 83.9 94.8 75.0 83.7 93.2 89.5 SE-GUI (Yuan et al., 2025) 7B - - - - - - 88.2 - - - - - - 90.3 GUI-G2 (Tang et al., 2025) 7B 96.7 90.8 95.9 88.6 90.9 86.9 92.0 98.3 91.9 95.4 89.3 94.0 87.7 93.3

Ours HyperClick

3B 96.7 83.9 92.8 80.7 88.7 83.5 88.5 98.6 86.3 95.4 90.6 82.2 84.7 90.6 7B 95.6 91.7 93.8 82.9 92.2 88.4 91.5 98.3 93.4 96.9 85.7 96.2 86.7 93.7

- Table 2: GUI grounding accuracy on the ScreenSpot-Pro (Li et al., 2025) benchmark over the CAD, Development, Creative, Scientific, Office, and OS sub-tasks. Bold and underline indicate the best and second-best results.

CAD Development Creative Scientific Office OS

Model Size

Avg. Text Icon Text Icon Text Icon Text Icon Text Icon Text Icon

General Models GPT-4o (OpenAI, 2024) - 2.0 0.0 1.3 0.0 1.0 0.0 2.1 0.0 1.1 0.0 0.0 0.0 0.8 Claude (Anthropic, 2024) - 14.5 3.7 22.0 3.9 25.9 3.4 33.9 15.8 30.1 16.3 11.0 4.5 17.1 Qwen2.5-VL (Bai et al., 2025) 7B 16.8 1.6 46.8 4.1 35.9 7.7 49.3 7.3 52.5 20.8 37.4 6.7 26.8

GUI-specific Models (SFT) CogAgent (Hong et al., 2024) 18B 7.1 3.1 14.9 0.7 9.6 0.0 22.2 1.8 13.0 0.0 5.6 0.0 7.7 SeeClick (Cheng et al., 2024) 9.6B 2.5 0.0 0.6 0.0 1.0 0.0 3.5 0.0 1.1 0.0 2.8 0.0 1.1 ShowUI (Lin et al., 2025) 2B 2.5 0.0 16.9 1.4 9.1 0.0 13.2 7.3 15.3 7.5 10.3 2.2 7.7 Aria-UI (Yang et al., 2024a) 25.3B 7.6 1.6 16.2 0.0 23.7 2.1 27.1 6.4 20.3 1.9 4.7 0.0 11.3 UGround (Gou et al., 2025) 7B 14.2 1.6 26.6 2.1 27.3 2.8 31.9 2.7 31.6 11.3 17.8 0.0 16.5 UGround-V1 (Gou et al., 2025) 7B 15.8 1.2 51.9 2.8 47.5 9.7 57.6 14.5 60.5 13.2 38.3 7.9 45.2 OS-Atlas (Wu et al., 2024) 7B 12.2 4.7 33.1 1.4 28.8 2.8 37.5 7.3 33.9 5.7 27.1 4.5 18.9 UI-TARS (Qin et al., 2025) 7B 20.8 9.4 58.4 12.4 50.0 9.1 63.9 31.8 63.3 20.8 30.8 16.9 35.7 TongUI (Zhang et al., 2025a) 7B 17.3 9.4 40.9 3.5 31.3 7.0 50.7 12.7 45.8 13.2 28.0 6.7 24.7 GUI-Actor (Wu et al., 2025) 7B - - - - - - - - - - - - 40.7 JEDI (Xie et al., 2025) 7B 38.0 14.1 42.9 11.0 50.0 11.9 72.9 25.5 75.1 47.2 33.6 16.9 39.5

GUI-specific Models (RL)

UI-R1 (Lu et al., 2025) 3B 11.2 6.3 22.7 4.1 27.3 3.5 42.4 11.8 32.2 11.3 13.1 4.5 17.8 UI-R1-E (Lu et al., 2025) 3B 37.1 12.5 46.1 6.9 41.9 4.2 56.9 21.8 65.0 26.4 32.7 10.1 33.5 GUI-R1 (Luo et al., 2025) 7B 23.9 6.3 49.4 4.8 38.9 8.4 55.6 11.8 58.7 26.4 42.1 16.9 31.3 InfiGUI-R1 (Liu et al., 2025b) 3B 33.0 14.1 51.3 12.4 44.9 7.0 58.3 20.0 65.5 28.3 43.9 12.4 35.7

- GUI-G1 (Zhou et al., 2025) 3B 39.6 9.4 50.7 10.3 36.6 11.9 61.8 30.0 67.2 32.1 23.5 10.6 37.1 SE-GUI (Yuan et al., 2025) 7B 51.3 42.2 68.2 19.3 57.6 9.1 75.0 28.2 78.5 43.4 49.5 25.8 47.3
- GUI-G2 (Tang et al., 2025) 7B 55.8 12.5 68.8 17.2 57.1 15.4 77.1 24.5 74.0 32.7 57.9 21.3 47.5 Ours

3B 43.7 23.5 62.4 20.0 50.5 12.6 55.6 30.0 63.9 37.8 41.1 20.2 41.3 7B 51.3 20.3 70.2 22.1 57.6 20.3 76.4 30.9 70.1 30.2 56.1 22.5 48.2

HyperClick

- Table 3: GUI grounding accuracy on the MMBench-GUI (Wang et al., 2025) benchmark over the Windows, MacOS, Linux, iOS, Android, and Web sub-tasks. Bold and underline indicate the best and second-best results.

Windows MacOS Linux iOS Android Web

Model Size

Avg. Basic Adv. Basic Adv. Basic Adv. Basic Adv. Basic Adv. Basic Adv.

General Models GPT-4o (OpenAI, 2024) - 1.5 1.1 8.7 4.3 1.1 1.0 5.1 3.3 2.5 1.4 3.2 2.9 2.9 Claude (Anthropic, 2024) - 1.5 0.7 12.5 7.5 1.1 0.0 13.7 10.6 1.4 1.4 3.2 2.3 4.7 Qwen-Max-VL (Bai et al., 2023) - 43.9 36.8 58.8 56.1 53.9 30.1 77.4 59.1 79.5 70.1 74.8 58.8 58.0 Qwen2.5-VL (Bai et al., 2025) 7B 31.4 16.5 31.3 22.0 21.5 12.2 66.6 55.2 35.1 35.2 40.3 32.5 33.9 InternVL3 (Zhu et al., 2025) 72B 70.1 42.6 75.7 52.3 59.2 41.3 93.6 80.6 92.7 78.6 90.7 65.9 72.2

GUI-specific Models (SFT)

ShowUI (Lin et al., 2025) 2B 9.2 4.4 24.1 10.4 25.1 11.7 29.0 19.7 17.4 8.7 22.9 12.7 16.0 OS-Atlas (Wu et al., 2024) 7B 36.9 18.8 44.4 21.7 31.4 13.3 74.8 48.8 69.6 46.8 61.3 35.4 41.4 Aguvis (Xu et al., 2024) 7B 37.3 21.7 48.1 33.3 33.5 25.0 67.5 65.2 61.0 51.0 61.6 45.5 45.7 UGround-V1 (Gou et al., 2025) 7B 66.8 39.0 71.3 48.6 56.5 31.1 92.7 70.9 93.5 71.0 88.7 64.6 65.7 UI-TARS (Qin et al., 2025) 72B 78.6 51.8 80.3 62.7 68.6 51.5 90.8 81.2 93.0 80.0 88.1 68.5 74.3

Ours HyperClick

3B 73.8 45.6 80.3 52.9 66.5 35.7 91.4 72.7 92.4 74.9 89.1 60.1 71.4 7B 82.3 61.4 82.9 67.1 66.5 48.0 94.0 82.1 95.8 85.1 93.2 85.1 79.6

ECEs of 0.589, 0.481, and 0.631. In contrast, HyperClick-7B reduces ECE to 0.249 and yields visibly smaller alignment gaps across most bins, showing that its verbalized confidence better tracks empirical grounding accuracy.

#### 4.4 Ablation Study

We conducted an ablation study on ScreenSpotV2 (Cheng et al., 2024) to verify the key components of HyperClick.

Reward Mechanism. The results in Table 4 demonstrate the need to combine format, correctness, and confidence alignment rewards. Using only the format reward yields relatively limited improvements (89.3%), and using only the confidence alignment reward (2.1%) causes reward hacking, where the policy model is optimized to predict only incorrect answers with confidence of 0. Moreover, we encourage the policy model to express confidence rounded to three decimal places and constrain it in Rformat to avoid overly long completions. This strict format verification is crucial for training stability and convergence, improving accuracy from 91.0% to 93.7%. In general, the combination of format, correctness, and confidence alignment rewards has the best performance. This validates our motivation that confidence alignment acts as an auxiliary uncertainty signal, discouraging overconfident errors and reinforcing reliable predictions.

Alignment Formulation. We further examine the sensitivity of HyperClick to the specific confidence-alignment formulation in Table 5. Following the principle that Rconfidence should be maxi-

- Table 4: Ablation study of reward configurations. * denotes reward hacking.

Rformat Rcorrectness Rconfidence Acc(%) ✓ 89.3

✓ 92.3 ✓ 2.1∗ ✓ ✓ 92.3 ✓ ✓ 91.0 ✓ ✓ ✓ 93.7

- Table 5: Ablation study of confidence reward formulations. We denote C¯ = C(ˆx,yˆ) and ∆ = |cˆ− C¯|.

Type Formulation Acc(%)

– – 92.3 l1 1 − ∆ 93.2 l2 1 − ∆2 93.7 KL exp −KL(C¯ ∥ cˆ) 93.6

- Table 6: Ablation study of confidence modeling.

###### Table 7: Ablation of baseline.

α Acc(%)

0 92.7 1/2 93.1 1/4 93.7 1/6 92.1

Model Acc(%) Qwen2.5-VL 88.8

- HyperClick 93.7 MiMo-VL 91.4

- HyperClick 94.0

mized when the predicted confidence cˆmatches the spatial confidence target C(ˆx,yˆ), we compare l1, l2, and KL variants. All three formulations improve over the no-confidence baseline, showing that the benefit does not depend on a single distance function. The l2 formulation performs best (93.7%) and is therefore used as the default, while KL achieves a comparable 93.6%.

#### Confidence Modeling. Table 6 investigates the

(a) UI-TARS-7B

(b) UI-TARS-1.5

(c) Qwen2.5-VL-7B

(d) HyperClick-7B

1.0

1.0

1.0

1.0

ECE = 0.589

ECE = 0.481

ECE = 0.631

ECE = 0.249

0.8

0.8

0.8

0.8

Accuracy

0.6

0.6

0.6

0.6

0.4

0.4

0.4

0.4

0.2

0.2

0.2

0.2

0.0

0.0

0.0

0.0

0.0 0.2 0.4 0.6 0.8 1.0

0.0 0.2 0.4 0.6 0.8 1.0

0.0 0.2 0.4 0.6 0.8 1.0

0.0 0.2 0.4 0.6 0.8 1.0

Confidence

Confidence

Confidence

Confidence

Perfect

Accuracy

Gap

| |
|---|

| |
|---|

- Figure 3: Reliability diagrams on ScreenSpot-Pro. Blue bars are the per-bin accuracy, red hatched bars are the gap to the perfect-alignment diagonal (green dashed), and the ECE value is shown in each panel. (a) UI-TARS-7B, (b) UI-TARS-1.5, and (c) Qwen2.5-VL-7B are severely overconfident, whereas (d) HyperClick-7B substantially reduces ECE and shows smaller alignment gaps across most confidence bins.

effect of the adaptive variance factor α. Without the bounded spatial confidence target (α=0), only binary confidence is used for confidence alignment. Therefore, when α=0, the confidence alignment reward is represented as:

Rconfidence = 1 − (ˆc − p∈ˆb)2. (8) The policy model reaches 92.7%, which is weaker than the Gaussian-target variants but still demonstrates the effectiveness of SCRL. We set α according to the Gaussian 3σ principle: in the x direction, k · σx = 12(x2 − x1), where k ∈ {1,2,3} gives α ∈ {12, 14, 61}. Both too large (α = 12) and too small (α = 16) variances are suboptimal: the former makes the target too diffuse across the box and weakens center-sensitive supervision, while the latter over-concentrates confidence near the center and becomes too strict for minor deviations. α = 14 provides the best concentration-spread trade-off and highest precision (93.7%).

Extension to other baselines. As shown in Table 7, we further extend HyperClick to MiMoVL (Xiaomi, 2025), a strong general-purpose VLM. With our training framework, MiMo-VL improves from 91.4% to 94.0%, demonstrating that HyperClick serves as a general training paradigm for GUI grounding, confirming the generality and scalability of our approach across various models.

#### 4.5 Visualization

To better understand confidence alignment, Figure 4 visualizes HyperClick’s predicted confidence distributions. We inject interface coordinates into the assistant generation and require the policy to continue outputting confidence for each click posi-

Instruction: Click search box.

Instruction: Click Privacy & security.

Instruction: Update apps.

Instruction: Check battery level.

Instruction: Check warnings.

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

Figure 4: Visualization of the confidence distribution output by HyperClick. We inject the coordinates on the interface into the assistant’s generation and enforce it to continue to output the confidence for the click position. The lighter the color, the higher the confidence value.

tion, yielding heatmaps over possible clicks. Confidence concentrates around ground-truth elements and stays low in irrelevant regions, matching the bounded spatial target design. Adaptive variance further adjusts the spread by element size: smaller UI elements yield tighter peaks, whereas larger ones produce broader heatmaps.

### 5 Conclusion

In this work, we address overconfidence in GUI grounding models, which undermines the reliability of autonomous GUI agents. We introduce HyperClick, a SCRL framework that augments grounding with explicit confidence alignment. By combining binary correctness reward with an l2 confidence alignment reward derived from a bounded spatial confidence target, HyperClick improves grounding accuracy while producing better-aligned confidence estimates. Extensive experiments show strong accuracy and confidence alignment, suggesting a path toward trustworthy GUI grounding in broader multimodal agents.

### Limitation

Although the effect of the confidence alignment mechanism proposed in this work has been verified, it has not been extended to GUI planning tasks. We believe that the reliability of planning is even more critical for the overall success of GUI automation, since inaccurate or overconfident planning decisions can propagate errors across multiple steps and ultimately lead to task failure. In future work, we plan to investigate how confidence alignment can be incorporated into planning modules, enabling agents to not only ground actions reliably but also expose uncertainty during high-level decisions throughout complex multi-step interactions.

### Ethics Consideration

This research focuses on building a policy model for reliable GUI grounding. The data used are obtained by synthesizing or reprocessing previously released datasets, with all datasets or benchmarks properly cited. In this paper, there are no discrimination, bias, or fairness issues that need to be addressed. In addition, our models are not expected to generate potentially harmful content. To ensure reproducibility, we provide all experimental and data details in Section 4 and the corresponding appendices. We will release the source code and model checkpoints to support reproducibility.

### References

Anthropic. 2024. Introducing computer use, a new claude 3.5 sonnet, and claude 3.5 haiku.

Chongyang Bai, Xiaoxue Zang, Ying Xu, Srinivas Sunkara, Abhinav Rastogi, Jindong Chen, and 1 others. 2021. Uibert: Learning generic multimodal representations for ui understanding. arXiv preprint arXiv:2107.13731.

Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. 2023. Qwen-vl: A versatile vision-language model for understanding, localization, text reading, and beyond. arXiv preprint arXiv:2308.12966.

Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, and 1 others. 2025. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923.

Zhe Chen, Weiyun Wang, Yue Cao, Yangzhou Liu, Zhangwei Gao, Erfei Cui, Jinguo Zhu, Shenglong Ye, Hao Tian, Zhaoyang Liu, and 1 others. 2024. Expanding performance boundaries of open-source

multimodal models with model, data, and test-time scaling. arXiv preprint arXiv:2412.05271.

Kanzhi Cheng, Qiushi Sun, Yougang Chu, Fangzhi Xu, Li YanTao, Jianbing Zhang, and Zhiyong Wu. 2024. SeeClick: Harnessing GUI grounding for advanced visual GUI agents. In Proceedings of the Annual Meeting of the Association for Computational Linguistics, pages 9313–9332.

Tri Dao. 2023. Flashattention-2: Faster attention with better parallelism and work partitioning. arXiv preprint arXiv:2307.08691.

Shrey Desai and Greg Durrett. 2020. Calibration of pre-trained transformers. arXiv preprint arXiv:2003.07892.

Zhikai Ding, Shiyu Ni, and Keping Bi. 2025. Do lvlms know what they know? a systematic study of knowledge boundary perception in lvlms. arXiv preprint arXiv:2508.19111.

Yichao Fu, Xuewei Wang, Yuandong Tian, and Jiawei Zhao. 2025. Deep think with confidence. arXiv preprint arXiv:2508.15260.

Carl Friedrich Gauss. 1809. Theoria motus corporum coelestium in sectionibus conicis solem ambientium auctore Carolo Friderico Gauss. sumtibus Frid. Perthes et IH Besser.

Carl Friedrich Gauss. 1877. Theoria motus corporum coelestium in sectionibus conicis solem ambientium, volume 7. FA Perthes.

Boyu Gou, Ruohan Wang, Boyuan Zheng, Yanan Xie, Cheng Chang, Yiheng Shu, Huan Sun, and Yu Su. 2025. Navigating the digital world as humans do: Universal visual grounding for GUI agents. In Proceedings of the International Conference on Learning Representations.

Chuan Guo, Geoff Pleiss, Yu Sun, and Kilian Q Weinberger. 2017. On calibration of modern neural networks. In Proceedings of the IEEE International Conference on Machine Learning, pages 1321–1330. PMLR.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, and 1 others. 2025a. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948.

Dong Guo, Faming Wu, Feida Zhu, Fuxing Leng, Guang Shi, Haobin Chen, Haoqi Fan, Jian Wang, Jianyu Jiang, Jiawei Wang, and 1 others. 2025b. Seed1. 5-vl technical report. arXiv preprint arXiv:2505.07062.

Kaiming He, Georgia Gkioxari, Piotr Dollár, and Ross Girshick. 2017. Mask r-cnn. In Proceedings of the IEEE International Conference on Computer Vision, pages 2961–2969.

Wenyi Hong, Weihan Wang, Qingsong Lv, Jiazheng Xu, Wenmeng Yu, Junhui Ji, Yan Wang, Zihan Wang, Yuxiao Dong, Ming Ding, and 1 others. 2024. Cogagent: A visual language model for gui agents. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14281–14290.

Ziwei Ji, Nayeon Lee, Rita Frieske, Tiezheng Yu, Dan Su, Yan Xu, Etsuko Ishii, Ye Jin Bang, Andrea Madotto, and Pascale Fung. 2023a. Survey of hallucination in natural language generation. ACM computing surveys, 55(12):1–38.

Ziwei Ji, Tiezheng Yu, Yan Xu, Nayeon Lee, Etsuko Ishii, and Pascale Fung. 2023b. Towards mitigating llm hallucination via self reflection. In Findings of the Association for Computational Linguistics: EMNLP 2023, pages 1827–1843.

Adam Tauman Kalai, Ofir Nachum, Santosh S Vempala, and Edwin Zhang. 2025. Why language models hallucinate. arXiv preprint arXiv:2509.04664.

Raghav Kapoor, Yash Parag Butala, Melisa Russak, Jing Yu Koh, Kiran Kamble, Waseem AlShikh, and Ruslan Salakhutdinov. 2024. Omniact: A dataset and benchmark for enabling multimodal generalist autonomous agents for desktop and web. In Proceedings of the European Conference on Computer Vision, pages 161–178. Springer.

Kaixin Li, Ziyang Meng, Hongzhan Lin, Ziyang Luo, Yuchen Tian, Jing Ma, Zhiyong Huang, and Tat-Seng Chua. 2025. Screenspot-pro: Gui grounding for professional high-resolution computer use. arXiv preprint arXiv:2504.07981.

Yang Li, Gang Li, Luheng He, Jingjie Zheng, Hong Li, and Zhiwei Guan. 2020. Widget captioning: Generating natural language description for mobile user interface elements. arXiv preprint arXiv:2010.04295.

Kevin Qinghong Lin, Linjie Li, Difei Gao, Zhengyuan Yang, Shiwei Wu, Zechen Bai, Stan Weixian Lei, Lijuan Wang, and Mike Zheng Shou. 2025. Showui: One vision-language-action model for gui visual agent. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 19498– 19508.

Stephanie Lin, Jacob Hilton, and Owain Evans. 2022. Teaching models to express their uncertainty in words. arXiv preprint arXiv:2205.14334.

Xinyi Liu, Xiaoyi Zhang, Ziyun Zhang, and Yan Lu. 2025a. Ui-e2i-synth: Advancing gui grounding with large-scale instruction synthesis. arXiv preprint arXiv:2504.11257.

Yuhang Liu, Pengxiang Li, Congkai Xie, Xavier Hu, Xiaotian Han, Shengyu Zhang, Hongxia Yang, and Fei Wu. 2025b. Infigui-r1: Advancing multimodal gui agents from reactive actors to deliberative reasoners. arXiv preprint arXiv:2504.14239.

Jonathan Long, Evan Shelhamer, and Trevor Darrell. 2015. Fully convolutional networks for semantic segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 3431–3440.

Zhengxi Lu, Yuxiang Chai, Yaxuan Guo, Xi Yin, Liang Liu, Hao Wang, Han Xiao, Shuai Ren, Guanjing Xiong, and Hongsheng Li. 2025. Ui-r1: Enhancing efficient action prediction of gui agents by reinforcement learning. arXiv preprint arXiv:2503.21620.

Run Luo, Lu Wang, Wanwei He, and Xiaobo Xia. 2025. Gui-r1: A generalist r1-style vision-language action model for gui agents. arXiv preprint arXiv:2504.10458.

Donald MacKenzie. 1988. The history of statistics: the measurement of uncertainty before 1900 by stephen m. stigler. Technology and Culture, 29(2):299–300.

Potsawee Manakul, Adian Liusie, and Mark JF Gales. 2023. Selfcheckgpt: Zero-resource black-box hallucination detection for generative large language models. arXiv preprint arXiv:2303.08896.

Mahdi Pakdaman Naeini, Gregory F Cooper, and Milos Hauskrecht. 2015. Obtaining well calibrated probabilities using bayesian binning. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 29.

Shravan Nayak, Xiangru Jian, Kevin Qinghong Lin, Juan A Rodriguez, Montek Kalsi, Rabiul Awal, Nicolas Chapados, M Tamer Özsu, Aishwarya Agrawal, David Vazquez, and 1 others. 2025. Ui-vision: A desktop-centric gui benchmark for visual perception and interaction. arXiv preprint arXiv:2503.15661.

Dang Nguyen, Jian Chen, Yu Wang, Gang Wu, Namyong Park, Zhengmian Hu, Hanjia Lyu, Junda Wu, Ryan Aponte, Yu Xia, and 1 others. 2024. Gui agents: A survey. arXiv preprint arXiv:2412.13501.

William L Oberkampf, Sharon M DeLand, Brian M Rutherford, Kathleen V Diegert, and Kenneth F Alvin. 2002. Error and uncertainty in modeling and simulation. Reliability Engineering & System Safety, 75(3):333–357.

OpenAI. 2024. Hello gpt-4o.

Yujia Qin, Yining Ye, Junjie Fang, Haoming Wang, Shihao Liang, Shizuo Tian, Junda Zhang, Jiahao Li, Yunxin Li, Shijue Huang, and 1 others. 2025. Uitars: Pioneering automated gui interaction with native agents. arXiv preprint arXiv:2501.12326.

Joseph Redmon, Santosh Divvala, Ross Girshick, and Ali Farhadi. 2016. You only look once: Unified, real-time object detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 779–788.

Shaoqing Ren, Kaiming He, Ross Girshick, and Jian Sun. 2015. Faster r-cnn: Towards real-time object detection with region proposal networks. volume 28.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. 2017. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, and 1 others. 2024. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300.

Haozhan Shen, Peng Liu, Jingcheng Li, Chunxin Fang, Yibo Ma, Jiajia Liao, Qiaoli Shen, Zilun Zhang, Kangjia Zhao, Qianqian Zhang, and 1 others. 2025. Vlm-r1: A stable and generalizable r1style large vision-language model. arXiv preprint arXiv:2504.07615.

Qiushi Sun, Kanzhi Cheng, Zichen Ding, Chuanyang Jin, Yian Wang, Fangzhi Xu, Zhenyu Wu, Chengyou Jia, Liheng Chen, Zhoumianze Liu, and 1 others. 2024. Os-genesis: Automating gui agent trajectory construction via reverse task synthesis. arXiv preprint arXiv:2412.19723.

Fei Tang, Zhangxuan Gu, Zhengxi Lu, Xuyang Liu, Shuheng Shen, Changhua Meng, Wen Wang, Wenqi Zhang, Yongliang Shen, Weiming Lu, and 1 others. 2025. Gui-g2: Gaussian reward modeling for gui grounding. arXiv preprint arXiv:2507.15846.

Gemini Team, Petko Georgiev, Ving Ian Lei, Ryan Burnell, Libin Bai, Anmol Gulati, Garrett Tanzer, Damien Vincent, Zhufeng Pan, Shibo Wang, and 1 others. 2024. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. arXiv preprint arXiv:2403.05530.

Kimi Team, Angang Du, Bohong Yin, Bowei Xing, Bowen Qu, Bowen Wang, Cheng Chen, Chenlin Zhang, Chenzhuang Du, Chu Wei, and 1 others. 2025. Kimi-vl technical report. arXiv preprint arXiv:2504.07491.

Katherine Tian, Eric Mitchell, Allan Zhou, Archit Sharma, Rafael Rafailov, Huaxiu Yao, Chelsea Finn, and Christopher D Manning. 2023. Just ask for calibration: Strategies for eliciting calibrated confidence scores from language models fine-tuned with human feedback. arXiv preprint arXiv:2305.14975.

Jianqiang Wan, Sibo Song, Wenwen Yu, Yuliang Liu, Wenqing Cheng, Fei Huang, Xiang Bai, Cong Yao, and Zhibo Yang. 2024. Omniparser: A unified framework for text spotting key information extraction and table recognition. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 15641–15653.

Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, and 1 others. 2024a. Qwen2vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191.

Shuai Wang, Weiwen Liu, Jingxuan Chen, Yuqi Zhou, Weinan Gan, Xingshan Zeng, Yuhan Che, Shuai Yu, Xinlong Hao, Kun Shao, and 1 others. 2024b. Gui agents with foundation models: A comprehensive survey. arXiv preprint arXiv:2411.04890.

Xuehui Wang, Zhenyu Wu, JingJing Xie, Zichen Ding, Bowen Yang, Zehao Li, Zhaoyang Liu, Qingyun Li, Xuan Dong, Zhe Chen, and 1 others. 2025. Mmbench-gui: Hierarchical multi-platform evaluation framework for gui agents. arXiv preprint arXiv:2507.19478.

Qianhui Wu, Kanzhi Cheng, Rui Yang, Chaoyun Zhang, Jianwei Yang, Huiqiang Jiang, Jian Mu, Baolin Peng, Bo Qiao, Reuben Tan, and 1 others. 2025. Gui-actor: Coordinate-free visual grounding for gui agents. arXiv preprint arXiv:2506.03143.

Zhiyong Wu, Zhenyu Wu, Fangzhi Xu, Yian Wang, Qiushi Sun, Chengyou Jia, Kanzhi Cheng, Zichen Ding, Liheng Chen, Paul Pu Liang, and 1 others. 2024. Os-atlas: A foundation action model for generalist gui agents.

LLM-Core-Team Xiaomi. 2025. Mimo-vl technical report. Preprint, arXiv:2506.03569.

Tianbao Xie, Jiaqi Deng, Xiaochuan Li, Junlin Yang, Haoyuan Wu, Jixuan Chen, Wenjing Hu, Xinyuan Wang, Yuhui Xu, Zekun Wang, and 1 others. 2025. Scaling computer-use grounding via user interface decomposition and synthesis. arXiv preprint arXiv:2505.13227.

Miao Xiong, Zhiyuan Hu, Xinyang Lu, Yifei Li, Jie Fu, Junxian He, and Bryan Hooi. 2023. Can llms express their uncertainty? an empirical evaluation of confidence elicitation in llms. arXiv preprint arXiv:2306.13063.

Yiheng Xu, Zekun Wang, Junli Wang, Dunjie Lu, Tianbao Xie, Amrita Saha, Doyen Sahoo, Tao Yu, and Caiming Xiong. 2024. Aguvis: Unified pure vision agents for autonomous gui interaction. arXiv preprint arXiv:2412.04454.

Yuhao Yang, Yue Wang, Dongxu Li, Ziyang Luo, Bei Chen, Chao Huang, and Junnan Li. 2024a. Aria-ui: Visual grounding for gui instructions. arXiv preprint arXiv:2412.16256.

Yuqing Yang, Ethan Chern, Xipeng Qiu, Graham Neubig, and Pengfei Liu. 2024b. Alignment for honesty. volume 37, pages 63565–63598.

Yuan Yao, Tianyu Yu, Ao Zhang, Chongyi Wang, Junbo Cui, Hongji Zhu, Tianchi Cai, Haoyu Li, Weilin Zhao, Zhihui He, and 1 others. 2024. Minicpm-v: A gpt-4v level mllm on your phone. arXiv preprint arXiv:2408.01800.

Wenwen Yu, Zhibo Yang, Jianqiang Wan, Sibo Song, Jun Tang, Wenqing Cheng, Yuliang Liu, and Xiang

Bai. 2025. Omniparser v2: Structured-points-ofthought for unified visual text parsing and its generality to multimodal large language models. arXiv preprint arXiv:2502.16161.

Xinbin Yuan, Jian Zhang, Kaixin Li, Zhuoxuan Cai, Lujian Yao, Jie Chen, Enguang Wang, Qibin Hou, Jinwei Chen, Peng-Tao Jiang, and 1 others. 2025. Enhancing visual grounding for gui agents via selfevolutionary reinforcement learning. arXiv preprint arXiv:2505.12370.

Bofei Zhang, Zirui Shang, Zhi Gao, Wang Zhang, Rui Xie, Xiaojian Ma, Tao Yuan, Xinxiao Wu, SongChun Zhu, and Qing Li. 2025a. Tongui: Building generalized gui agents by learning from multimodal web tutorials. arXiv preprint arXiv:2504.12679.

Chaoyun Zhang, Shilin He, Jiaxu Qian, Bowen Li, Liqun Li, Si Qin, Yu Kang, Minghua Ma, Guyue Liu, Qingwei Lin, and 1 others. 2024. Large language model-brained gui agents: A survey. arXiv preprint arXiv:2411.18279.

Jiaxin Zhang, Zhuohang Li, Kamalika Das, Bradley A Malin, and Sricharan Kumar. 2023. Sac3: reliable hallucination detection in black-box language models via semantic-aware cross-check consistency. arXiv preprint arXiv:2311.01740.

Shaojie Zhang, Ruoceng Zhang, Pei Fu, Shaokang Wang, Jiahui Yang, Xin Du, Shiqi Cui, Bin Qin, Ying Huang, Zhenbo Luo, and 1 others. 2025b. Btlui: Blink-think-link reasoning model for gui agent. arXiv preprint arXiv:2509.15566.

Zhong Zhang, Yaxi Lu, Yikun Fu, Yupeng Huo, Shenzhi Yang, Yesai Wu, Han Si, Xin Cong, Haotian Chen, Yankai Lin, and 1 others. 2025c. Agentcpm-gui: Building mobile-use agents with reinforcement finetuning. arXiv preprint arXiv:2506.01391.

Yuqi Zhou, Sunhao Dai, Shuai Wang, Kaiwen Zhou, Qinglin Jia, and Jun Xu. 2025. Gui-g1: Understanding r1-zero-like training for visual grounding in gui agents. arXiv preprint arXiv:2505.15810.

Jinguo Zhu, Weiyun Wang, Zhe Chen, Zhaoyang Liu, Shenglong Ye, Lixin Gu, Hao Tian, Yuchen Duan, Weijie Su, Jie Shao, and 1 others. 2025. Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models. arXiv preprint arXiv:2504.10479.

### A Appendix

#### A.1 Data Details

To provide a comprehensive grounding resource across diverse platforms, we construct a dataset containing 50K samples distributed across three representative domains: Mobile, Web, and Desktop. Training data is sampled from multiple public GUI datasets, including OS-Atlas (Wu et al., 2024), Widget Caption (Li et al., 2020), UI-Refexp (Bai et al., 2021), and OmniAct (Kapoor et al., 2024), together with in-house data. Each domain contains a balanced set of grounding instances that pair natural language commands with corresponding UI elements. The per-source statistics are shown in Table 8.

Table 8: Statistics and sources of the grounding dataset adopted in HyperClick.

Source OmniAct ShowUI-Web UI-Refexp Widgnt-Caption OS-Atlas In-House Size 119 19172 280 3672 26114 1664

To construct high-quality samples for RL, we first employ Qwen2.5-VL-7B (Bai et al., 2025) to generate raw data with the temperature set to 0, and identify cases where the model produces incorrect predictions. For each of these error cases, we then perform eight additional inferences with temperature 0.9 and extract the correctly predicted results as the final training data. In addition, prior to RL, we incorporate an equal number of correctly predicted samples from Stage 1 to provide a cold start. This initialization not only stabilizes the training but also helps the model adhere to the target output format: <point>[x,y]</point> <confidence>conf</confidence>, where conf is rounded to three decimal places.

#### A.2 Evaluation Prompts

In this section, we detail the replicated evaluation prompts in ScreenSpot-Pro (Li et al., 2025). We follow the instructions they originally provided to reproduce and analyze the experimental results. The prompts are shown as follows:

GPT-4o’s Prompt

Locate the UI element most related to the instruction {problem} on the screenshot. Output only a JSON in the format [{“point_2d”: [...]}].

Seed-VL’s Prompt

Locate the UI element most related to the instruction {problem} on the screenshot. Output only a JSON in the format [{“point_2d”: [...]}].

Qwen2.5-VL’s Prompt

Locate the UI element most related to the instruction {problem} on the screenshot. Output only a JSON in the format [{“point_2d”: [...], “label”: ... }].

KiMi-VL’s Prompt

Point to the UI element most related to the instruction {problem} on the screenshot.

MiMo-VL’s Prompt

Locate the UI element most related to the instruction {problem} on the screenshot. Output a JSON format [{“bbox_2d”: [...], “label”: ...}]./no_think

UI-TARS’ and UI-TARS-1.5’s Prompt

Point to the element related to the instruction {problem} on the screenshot.

Due to UI-TARS (Qin et al., 2025) and UITARS-1.5 (Qin et al., 2025) being trained with a large amount of GUI-specific data, the ability to follow instructions is relatively poor. To prompt such models to generate verbalized confidence in their predictions, we adopt a multi-round conversation to output confidence for their answer. Specifically, policy models use the above prompts for GUI grounding in the first round and in the second round, generate the verbalized confidence of the prediction according to the prompt below:

Confidence Prompt

Output only a float number ranging from 0 to 1, representing your confidence with your provided answer, without any format.

#### A.3 Training Dynamics

Figure 5 tracks two GRPO training signals over the course of RL fine-tuning for HyperClick-3B and

HyperClick-7B: the mean reward and the per-group reward standard deviation. The mean reward rises steadily and plateaus, indicating that the policy progressively produces grounding outputs that satisfy the self-critique reward. The reward standard deviation stays well above zero throughout training, showing that GRPO continues to sample diverse rollouts within each prompt group and thus retains a usable advantage signal rather than collapsing to a degenerate policy. Together, the two curves confirm that the self-critiqued objective drives stable optimization without sacrificing exploration.

1.0

0.8

0.6

Reward

0.4

Accuracy reward

0.2

Confidence reward

Format reward

0.0

0 500 1000 1500 2000 2500 3000 Steps

(a) Mean reward over training steps.

Reward std

0.6

0.5

Within-grouprewardstd

0.4

0.3

0.2

0.1

0.0

0 500 1000 1500 2000 2500 3000 Steps

(b) Reward standard deviation over training steps.

###### Figure 5: GRPO training dynamics for HyperClick-

- 3B and HyperClick-7B. (a) The mean reward rises and stabilizes, indicating that the policy increasingly satisfies the self-critique reward. (b) The per-group reward standard deviation stays non-trivial, showing that GRPO retains rollout diversity and a usable advantage signal throughout training.

maintain low variance across different sampling scales, with the larger 7B model showing slightly more stable outputs. This suggests that the confidence estimation of HyperClick is stable under repeated sampling.

Table 9: Stability evaluation of the model for the same prediction.

Variance

Model

10 50 100 500 1000 1581

HyperClick-3B 0.020 0.028 0.023 0.020 0.020 0.020 HyperClick-7B 0.014 0.020 0.020 0.019 0.019 0.019

#### A.5 More Evaluation Benchmarks and

Detailed Experimental Results In this section, we present additional benchmarks and experimental results used in this work.

CAGUI is a Chinese benchmark for mobile GUI grounding. It emphasizes the grounding of textual elements and functional operations within Chineselanguage applications. Detailed experimental results and comparisons with baselines are shown in

- Table 10. UI-I2E-Bench introduces implicit instructions

that require both semantic understanding and spatial reasoning. Highlights the limitations of direct grounding and encourages models to adopt more sophisticated reasoning. Detailed experimental results and comparisons with baselines are shown in

- Table 11. UI-Vision evaluates the generalization of cross-

applications in diverse desktop environments. By incorporating previously unseen applications, it tests the model’s robustness and adaptability. Detailed experimental results and comparisons with baselines are shown in Table 12.

#### A.4 Stability of Confidence

To evaluate the stability of HyperClick’s confidence, we verify whether the model gives similar confidence estimates for the same answer. We first let HyperClick predict the coordinates without sampling. Then, we inject the coordinates into the assistant’s generation and instruct it to continue outputting confidence at a temperature of 1.0 for 8 times. As shown in Table 9, we report the mean variance for different sample sizes. The results indicate that both HyperClick-3B and HyperClick-7B

- Table 10: GUI grounding accuracy on the CAGUI (Zhang et al., 2025c) benchmark over the Fun2Point, Text2Point, and Bbox2Text sub-tasks. Bold and underline indicate the best and second-best results.

Model Size Fun2Point Text2Point Avg. General Models

GPT-4o (OpenAI, 2024) - 22.1 19.9 21.0 Qwen2.5-VL (Bai et al., 2025) 7B 59.8 59.3 59.6 InternVL2.5 (Chen et al., 2024) 8B 17.2 24.2 20.7

GUI-specific Models (SFT) OS-Genesis (Sun et al., 2024) 7B 8.3 5.8 7.1 OS-Atlas (Wu et al., 2024) 7B 53.6 60.7 57.2 Aguvis (Xu et al., 2024) 7B 60.8 76.5 68.7 UI-TARS (Qin et al., 2025) 7B 56.8 66.7 61.8

GUI-specific Models (RL) AgentCPM-GUI (Zhang et al., 2025c) 8B 79.1 76.5 77.8 Ours HyperClick

3B 80.9 81.2 81.0 7B 82.7 83.1 82.9

- Table 11: GUI grounding accuracy on the UI-I2E-Bench (Liu et al., 2025a) benchmark over the platforms of mobile, desktop, and web with various implicitness. Bold and underline indicate the best and second-best results.

Platform Implicitness

Model Size

Avg. Mobile Desktop Web Explicit Implicit

General Models Qwen2.5-VL (Bai et al., 2025) 7B 61.7 41.6 56.9 58.4 51.0 53.8

GUI-specific Models (SFT)

ShowUI (Lin et al., 2025) 2B 53.9 30.4 29.6 51.3 35.6 41.5 SeeClick (Cheng et al., 2024) 9.6B 37.2 15.8 18.2 37.1 19.9 26.4 Aguvis (Xu et al., 2024) 7B 60.3 47.6 45.1 61.1 48.4 53.2 OmniParser (Wan et al., 2024) - 67.6 45.5 30.8 54.3 52.4 53.1 OmniParser (Yu et al., 2025) - 69.4 42.4 40.7 57.0 53.5 54.8 OS-Atlas (Wu et al., 2024) 7B 68.1 48.9 52.2 63.2 55.8 58.6 UGround-V1 (Gou et al., 2025) 7B 73.5 65.7 70.8 81.3 63.6 70.3 UI-TARS (Qin et al., 2025) 7B 65.7 58.0 56.5 71.4 55.3 61.4 UI-I2E-VLM (Liu et al., 2025a) 7B 76.2 64.0 62.1 72.0 67.9 69.5

GUI-specific Models (RL) UI-R1 (Lu et al., 2025) 3B 67.8 46.2 58.1 67.9 52.8 58.5

Ours HyperClick

3B 77.9 59.0 81.0 81.1 66.1 71.8 7B 80.4 67.5 84.2 84.8 71.4 76.5

- Table 12: GUI grounding accuracy on the UI-Vision (Nayak et al., 2025) benchmark over the Education (Ed.), Browsers (Br.), Development (De.), Productivity (Pr.), Creativity (Cr.), and Entertainment (En.) subtasks. Bold and underline indicate the best and second-best results.

Setting Category

Model Size

Avg. Basic Functional Spatial Ed. Br. De. Pr. Cr. En.

General Models

GPT-4o (OpenAI, 2024) - 1.6 1.5 1.0 1.5 0.0 2.2 1.1 0.8 4.2 1.4 Gemini-1.5-pro (Team et al., 2024) - 0.8 0.3 0.6 0.5 0.6 0.9 0.5 0.4 0.0 0.6 Claude (Anthropic, 2024) - 9.5 7.7 7.6 6.1 9.8 8.0 9.4 7.7 8.3 8.3 Qwen2.5-VL (Wang et al., 2024a) 7B 1.2 0.8 0.5 0.5 0.0 1.2 0.9 0.5 1.0 0.9 InternVL2.5 (Chen et al., 2024) 8B 2.5 2.8 1.0 1.1 7.0 3.0 1.8 1.2 5.2 2.1 MiniCPM-V (Yao et al., 2024) 8B 7.1 5.3 1.5 3.0 16.8 5.4 3.8 2.1 13.0 4.3

GUI-specific Models (SFT) CogAgent (Hong et al., 2024) 9B 12.0 12.2 2.6 8.7 11.2 8.6 10.3 5.6 15.6 8.9 SeeClick (Cheng et al., 2024) 9.6B 9.4 4.7 2.1 4.2 13.3 7.3 4.3 4.0 11.0 5.4 AriaUI (Yang et al., 2024a) 25.3B 12.2 14.0 4.0 9.0 18.9 11.2 10.4 6.5 19.3 10.1 ShowUI (Lin et al., 2025) 2B 8.1 7.7 2.1 3.7 13.3 7.5 6.5 2.5 15.6 5.9 OS-Atlas (Wu et al., 2024) 7B 12.2 11.2 3.7 8.7 16.8 10.3 9.2 5.6 16.2 9.0 UGround-V1 (Nayak et al., 2025) 7B 15.4 17.1 6.3 10.4 28.7 17.5 12.2 8.6 18.2 12.9 Aguvis (Xu et al., 2024) 7B 17.8 18.3 5.1 13.1 30.8 17.1 12.1 9.6 24.0 13.7 UI-TARS (Qin et al., 2025) 7B 20.1 24.3 8.4 14.2 35.0 19.7 18.3 11.1 38.5 17.6

Ours HyperClick

3B 28.7 24.4 6.8 19.6 30.8 20.6 21.1 12.7 40.6 19.6 7B 35.3 32.1 11.0 24.3 47.6 26.5 27.1 18.3 50.0 25.7

