# arXiv:2606.03746v2[cs.CV]3Jun2026

[Figure 1]

June 4, 2026

## Qwen-Image-Flash: Beyond Objective Design

Tianhe Wu, Kun Yan, Zikai Zhou, Lihan Jiang, Jiahao Li, Jie Zhang, Kaiyuan Gao, Ningyuan Tang, Shengming Yin, Xiaoyue Chen, Xiao Xu, Yilei Chen, Yuxiang Chen, Yan Shu, Yixian Xu, Yanran Zhang, Zihao Liu, Zhendong Wang, Zekai Zhang, Deqing Li, Liang Peng, Yi Wang, Jingren Zhou, Chenfei Wu∗

{wutianhe.wth, fulai.hr}@alibaba-inc.com

### Abstract

Few-step distillation has become an effective strategy for accelerating advanced visual generative models, yet prior work has largely focused on distillation objectives. In this work, we revisit few-step distillation from a complementary perspective, focusing on the training recipe that critically shapes student performance. Using Qwen-Image-2.0 as a representative case, we systematically investigate three factors in unified text-to-image generation and instruction-guided image editing distillation: data composition, teacher guidance, and task mixture. Our empirical analysis reveals several non-obvious behaviors, which motivate the development of Qwen-Image-Flash. Overall, our results suggest that effective few-step distillation requires not only carefully designed objectives, but also principled organization of the broader training pipeline.

[Figure 2]

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

[Figure 18]

Figure 1: Qwen-Image-Flash examples. T2I and instruction-guided editing results with only 4 NFEs, showing unified few-step generation-editing capability.

∗Corresponding author

### Contents

- 1 Introduction 3
- 2 Preliminaries: Flow Matching and DMD 3

- 2.1 Flow Matching . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 4
- 2.2 DMD Objective . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 4

- 3 Data Composition Matters in T2I Distillation 4

- 3.1 Training Setup . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 4
- 3.2 T2I-Bench . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 5
- 3.3 Counterintuitive Effects of Data Diversity in T2I Distillation . . . . . . . . . . . . . . . . . 6

- 4 Stabilizing Complementary Teacher Guidance 6

- 4.1 Motivation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 7
- 4.2 Observation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 8
- 4.3 Step-wise Multi-teacher Guidance . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 8
- 4.4 Stable Distillation with Multi-teacher Guidance . . . . . . . . . . . . . . . . . . . . . . . . . 8

- 5 Joint Distillation for T2I Generation and Editing 9

- 5.1 Editing-Bench . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 9
- 5.2 Task-mixture Composition . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 10
- 5.3 Task-ratio Sensitivity in Unified Generation-Editing Distillation . . . . . . . . . . . . . . . 10
- 5.4 Editing Supervision Benefits T2I Generation . . . . . . . . . . . . . . . . . . . . . . . . . . 11

- 6 Discussion 11

- 6.1 Unsuccessful Attempts . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 11
- 6.2 Limitations and Future Work . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 12

- 7 Related Work 13

- 7.1 Few-step Distillation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 13
- 7.2 Benchmarks for Efficient Visual Generation and Editing . . . . . . . . . . . . . . . . . . . . 13

- 8 Conclusion 13 References 13 Appendix 16 A Evaluation Details 16

- A.1 System Prompts Used in Evaluation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 16
- A.2 T2I-Bench Hard Cases . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 17

### 1 Introduction

Visual generative models have moved beyond conventional text-to-image (T2I) systems and are increasingly developing into general-purpose visual foundation models (Esser et al., 2024; Song et al., 2026; Liu et al., 2026a; Mao et al., 2026; Song et al., 2026). Modern models can generate high-fidelity images from complex prompts, produce dense and structured visual text, leverage post-training to improve alignment and visual preference (Liu et al., 2026b), and support instruction-guided image editing within a unified framework (Zhao et al., 2026). These advances expand their applicability to content creation, graphic design, interactive editing, and multimodal applications.

However, the practical use of these models is still constrained by their sampling cost. Diffusion (Ho et al., 2020; Song et al., 2020) and flow-based (Lipman et al., 2022) visual generators typically synthesize images through iterative trajectories, requiring many function evaluations during inference and thereby incurring substantial latency and computation. This makes deployment challenging in latency-sensitive or resource-limited settings, including interactive image editing (Meng et al., 2022; Brooks et al., 2022), on-device generation (Li et al., 2023; Zhao et al., 2024), and large-scale visual content production (Azuma, 1997; Yin et al., 2025). Few-step distillation addresses this limitation by compressing the sampling behavior of a multi-step teacher into a student model that can generate with only a few steps.

Fast visual generation has been substantially advanced by the design of distillation objectives, including trajectory-level alignment (Geng et al., 2025), consistency training (Song et al., 2023), adversarial distillation (Sauer et al., 2024), and distribution matching (Yin et al., 2024b;a; Jiang et al., 2025). Nevertheless, when existing distillation methods are directly applied to large-scale visual generative models in broad and heterogeneous scenarios, such as text-centric rendering, a seemingly intuitive and conventional training recipe often falls short of the desired performance, as illustrated in Figure 2. This failure reminds us that the distillation objective is only part of the story, and that effective distillation must also account for the broader training recipe in which the objective is embedded.

This observation naturally shifts our attention from designing distillation objectives in isolation to understanding the broader training recipe that determines whether such objectives can be effective in practice. These considerations lead us to examine a more practical question: when distilling advanced visual generative models into few-step students, what training-time design choices matter beyond the distillation objective itself? We instantiate this study with Qwen-Image-2.0 (Zhao et al., 2026) and systematically analyze three key dimensions: data composition for T2I distillation, teacher guidance for leveraging teachers with different capabilities, and task mixture for joint T2I-editing distillation.

Our empirical analysis leads to three key findings. First, T2I distillation is highly sensitive to data composition: increasing diversity or using more target-specific data does not necessarily improve performance, whereas coherent data from a single category can transfer unexpectedly well. Second, transferring knowledge from teachers with complementary strengths across downstream tasks remains challenging. To this end, we propose a step-wise multi-teacher guidance strategy that combines their task-specific expertise while preserving training stability. Third, in joint T2I-editing distillation, the task mixture plays a decisive role, with the best unified performance achieved under a balanced T2I-to-editing data ratio (T2I:Edit). Together, these observations suggest that few-step distillation of modern visual generative models is shaped not only by the objective, but also by how data, teachers, and tasks are structured during training.

Building on these findings, we develop Qwen-Image-Flash, a unified few-step model for both T2I generation and instruction-guided image editing. As shown in Figure 1, Qwen-Image-Flash reduces the number of function evaluations (NFE) to only 4, while maintaining high visual quality and strong synthesis capabilities across diverse scenarios (e.g., poster generation). Rather than viewing few-step distillation as a matter of objective design alone, our work emphasizes the training recipe that enables advanced visual generative capabilities to be reliably transferred to efficient students. Qwen-Image-Flash thus embodies our central message: effective distillation must go beyond the objective.

### 2 Preliminaries: Flow Matching and DMD

We briefly review two components used in this work: flow matching (Lipman et al., 2022), a continuoustime framework for learning generative dynamics, and DMD (Yin et al., 2024a), which we adopt to distill a multi-step teacher into a few-step student.

#### 2.1 Flow Matching

Flow matching defines a transport process between data and noise by prescribing a probability path and then learning the velocity field along this path. Let x ∼ pdata denote a data point and let ϵ ∼ pnoise be an independent noise sample, where pnoise is typically set to N (0,I). In this work, following (Liu et al., 2023a; Geng et al., 2025), we use the linear path

zt = (1 − t)x + tϵ, t ∈ [0,1]. (1)

This path starts from the data distribution at t = 0 and reaches the noise distribution at t = 1. The condition c represents any side information used by the generative model, such as labels, text embeddings, or task-specific guidance signals.

Under the above interpolation, the velocity that moves a point along the path is ϵ − x. Therefore, flow matching trains a parameterized vector field vθ(zt, t, c) to predict this velocity:

ℓFM(θ) = Et,x,ϵ vθ(zt, t, c) − (ϵ − x) 2 . (2)

After training, samples are generated by initializing z1 from the noise prior and integrating the learned ODE from t = 1 back to t = 0. The generated sample is thus obtained as xθ = z1 + 1 0 vθ(zt, t, c) dt.

#### 2.2 DMD Objective

DMD is designed to distill a pretrained multi-step teacher into a conditional student generator Gθ. Given an input noise variable ϵ and condition c, the student produces a clean sample xθ = Gθ(ϵ, c). To compare the student with the teacher at noisy intermediate states, an additional independent noise sample ξ ∼ pnoise is drawn, and the student sample is perturbed through xt = (1 − t)xθ + tξ with t ∼ pt. At a high level, DMD encourages the conditional distribution induced by the student to approach that of the teacher. This can be written as the following KL objective:

ℓDMD(θ) ≜ DKL pstu(xθ | c) ∥ ptea(xθ | c) . (3)

Rather than optimizing this divergence directly, DMD uses a gradient estimator based on the difference between the score field of the student distribution and that of the teacher distribution:

∇θℓDMD(θ) = Eϵ,ξ,t ∇θxθ ⊺ sstu(xt, t, c) − sreal(xt, t, c) . (4)

Here, sstu is estimated using an auxiliary score network trained on samples generated by the student, while sreal is obtained from the pretrained teacher. The resulting update pushes the student toward regions where its noisy marginal score agrees with the teacher score across sampled noise levels.

### 3 Data Composition Matters in T2I Distillation

This section examines how the composition of distillation data shapes T2I student performance, with emphasis on both general image generation and challenging text-centric synthesis scenarios.

#### 3.1 Training Setup

We use Qwen-Image-2.0-Base (Zhao et al., 2026) as the multi-step teacher and distill it into a 4-NFE student with DMD (Yin et al., 2024a). The teacher is the pretrained base model and is not further enhanced by preference learning, reinforcement learning, or other post-training procedures, allowing us to focus on how different distillation data distributions affect the student.

We construct the distillation prompts with Qwen3 (Yang et al., 2025) across three representative categories: landscapes, portraits, and text-centric scenarios. Each category contains 20,000 diverse prompts. Based on these category-specific prompt sets, we design five training compositions with different levels of category coverage: landscape-only, portrait-only, text-centric-only, landscape-portrait, and mixed-category data containing all three categories. All students are trained under the same optimization protocol for 2,000 iterations using AdamW (Loshchilov &Hutter, 2017), so that performance differences can be attributed primarily to the choice of training data composition.

Table 1: Quantitative comparison of T2I distillation under different training data compositions. We evaluate 4-NFE students distilled with different category-specific and mixed-category training sets on landscape, portrait, and text-centric splits of T2I-Bench.

|Exp.<br><br>|Training data composition|# of training data<br><br>|Metrics|T2I-Bench Landscape Portrait Text-centric<br><br>|Average<br><br>|Rank|
|---|---|---|---|---|---|---|
|E1<br><br>|Landscape|20,000<br><br>|Gemini 3.1 Pro GPT 5.5|3.53 3.37 3.01<br><br>4.30 4.31 3.77<br><br><br>|3.30 4.13|3<br><br>|
|E2<br><br>|Portrait|20,000<br><br>|Gemini 3.1 Pro GPT 5.5<br><br>|3.56 3.57 3.12<br>4.35 4.34 3.76<br>|3.42 4.15<br><br>|1|
|E3|Text-centric<br><br>|20,000|Gemini 3.1 Pro GPT 5.5<br><br>|2.55 3.38 1.97<br>3.34 3.88 2.64<br>|2.63 3.29<br><br>|5|
|E4<br><br>|Landscape-Portrait|40,000<br><br>|Gemini 3.1 Pro GPT 5.5<br><br>|3.61 3.54 3.04<br>4.24 4.33 3.62<br>|3.40 4.06<br><br>|2|
|E5|Mixed-category<br><br>|60,000<br><br>|Gemini 3.1 Pro GPT 5.5|3.53 3.47 2.05<br><br>4.08 4.23 2.54<br><br><br>|3.02 3.62|4<br><br>|

Text-centric Mixed-category Landscape Landscape-Portrait Portrait

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

- Figure 2: Qualitative comparison of T2I distillation under different training data compositions. We compare students distilled with text-centric, mixed-category, landscape-only, landscape-portrait, and portrait-only training data across representative evaluation scenarios. The results show that text-centric or more diverse mixed-category data does not necessarily improve text rendering or overall visual quality. In contrast, students trained on coherent single-category data, such as landscape or portrait prompts, produce more faithful and visually stable results, suggesting stronger cross-category transfer and underscoring the importance of data composition in few-step distillation.

#### 3.2 T2I-Bench

To support a rigorous and systematic evaluation of few-step T2I generation, we introduce T2I-Bench, a challenging benchmark covering the same three categories used in our data-composition study. T2I-Bench contains 1,800 evaluation cases in total, with 600 samples for each category. We adopt Gemini 3.1 Pro and GPT 5.5 as automatic preference-based evaluators to assess the perceptual quality of generated images, where higher scores correspond to better visual fidelity and stronger alignment with human preference. Details of the evaluation are provided in the Appendix.

#### 3.3 Counterintuitive Effects of Data Diversity in T2I Distillation

- Table 1 summarizes the quantitative results under different T2I distillation data compositions. Rather than exhibiting a monotonic improvement with broader category coverage, the results reveal several counterintuitive patterns: data that appears more directly aligned with the target capability is not necessarily more effective, and increasing diversity can even degrade the distilled student.

Distillation performance is highly dependent on data distribution. The first observation is that T2I distillation is highly sensitive to the distribution of the training data. A particularly counterintuitive case is the text-centric-only setting (E3). Although this setting appears to provide the most direct supervision for text rendering, it achieves the lowest average performance among all evaluated configurations and degrades across all three evaluation categories. More notably, its weakness is not limited to out-of-domain cases: even on the text-centric split itself, E3 performs worse than the landscape-only and portrait-only settings (E1 and E2), which contain no explicit text-centric distillation data. This suggests that directly exposing the student to text-heavy samples does not automatically translate into stronger text rendering ability. Instead, such data may introduce optimization or distributional difficulties that reduce the overall effectiveness of knowledge transfer, as also reflected in the qualitative comparison in Figure 2.

The mixed-category setting (E5) reinforces this conclusion from another perspective. Although E5 uses the largest training set and covers all three prompt categories, it still fails to outperform the stronger single-category settings. In particular, adding text-centric samples into the mixture causes a clear drop on the text-centric benchmark relative to E1 and E2. This behavior contrasts with the common intuition from large-scale pretraining, where larger and more diverse datasets are often expected to improve coverage of the target distribution. In few-step distillation, however, the training data does not merely serve as broad coverage; it also determines how the teacher’s distributional guidance is exposed to a capacityand trajectory-limited student. As a result, simply mixing more heterogeneous categories can dilute or destabilize the transfer process. These findings indicate that distillation data must be selected with care, since certain data types, including seemingly relevant text-centric samples, can be ineffective or even harmful for transferring the teacher’s capabilities.

Takeaway 1: Training data selection is critical in T2I distillation; unlike pretraining, simply adding data to better cover the target distribution can be ineffective or even detrimental.

Coherent single-category data can support broad transfer. A second observation is that a coherent single-category distillation set can generalize well beyond its own domain. Both the landscape-only setting (E1) and the portrait-only setting (E2) perform strongly not only on their corresponding in-domain evaluation splits, but also on categories that are absent from their training data. This cross-category transfer is especially striking on the text-centric split: despite never using text-centric prompts during distillation, E1 and E2 both outperform the text-centric-only setting (E3) and the mixed-category setting (E5). These results suggest that effective few-step distillation does not require the training data to explicitly cover every downstream scenario. Instead, a clean and coherent training distribution may provide a more favorable interface for transferring general visual synthesis ability from the teacher, including capabilities that later emerge in challenging text-centric cases.

We further test whether combining two individually strong and generalizable categories yields additional gains. To this end, we construct the landscape-portrait setting (E4), which merges the two single-category datasets that already show strong transfer. However, E4 does not surpass the best single-category configuration, namely portrait-only distillation (E2). Despite using twice as many training samples and covering a broader visual range, E4 obtains a lower average score than E2. This result suggests that the benefit of a coherent training distribution can outweigh the apparent advantage of broader category coverage. In T2I distillation, therefore, more categories do not necessarily provide better supervision; they may instead weaken the consistency of the training signal and reduce the efficiency of student learning.

Takeaway 2: Single-category distillation can generalize beyond the training domain, whereas combining multiple generalizable categories may even impair student performance.

### 4 Stabilizing Complementary Teacher Guidance

In this section, we study how to effectively exploit multiple teachers with complementary downstream capabilities within the DMD framework (Yin et al., 2024a). We first show that naively replacing the base teacher with a task-specialized teacher can destabilize few-step distillation, despite the stronger

- (a) Task-specialized teacher guidance
- (b) Step-wise multi-teacher as guidance

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

Iteration 50 Iteration 200 Iteration 400 Iteration 600 Iteration 850

- Figure 3: Qualitative comparison of teacher guidance strategies during distillation. (a) Direct guidance from a task-specialized teacher can destabilize training, leading to progressive degradation in alignment and visual quality. (b) Step-wise multi-teacher guidance maintains sample fidelity and layout consistency throughout distillation, yielding better-aligned generations.

downstream performance of the specialized teacher. We then introduce a simple yet effective step-wise multi-teacher guidance strategy that uses the pretrained base teacher as a stable distributional anchor while selectively incorporating task-specialized guidance during distillation. This design allows the student to benefit from complementary teacher expertise without sacrificing optimization stability.

#### 4.1 Motivation

Few-step visual generators are expected to operate reliably across diverse downstream scenarios, including landscape synthesis, portrait generation, and text-centric image generation. However, teacher models often exhibit non-uniform strengths across these tasks. A straightforward solution is to use the strongest teacher for each target downstream task as the sole guidance model. However, we find that this strategy is not consistently reliable in few-step distillation. Task-specialized teachers can induce sharper and more concentrated task-specific distributions, which may increase the mismatch between the teacher score field and the distribution currently represented by the student. This mismatch is particularly challenging for few-step students, since their limited sampling trajectories provide less flexibility to gradually approximate highly specialized teacher behavior.

#### 4.2 Observation

To examine whether task-specialized teachers can be directly used for few-step distillation, we replace the pretrained base teacher with a teacher that performs better on a target downstream subset. The experiment follows the same protocol as Section 3, using the same training data composition and optimization hyperparameters.

Direct specialized-teacher guidance destabilizes distillation. As shown in Figure 3 (a), directly distilling from a task-specialized teacher leads to clear optimization instability. Although the student initially benefits from some task-specific improvements, generation quality deteriorates as training continues. The resulting samples exhibit structural misalignment, reduced visual fidelity, and weaker semantic consistency. This behavior suggests that stronger downstream teacher performance does not automatically translate into better few-step distillation guidance. Under standard DMD optimization, the distribution induced by a specialized teacher may be harder for a few-step student to approximate. We hypothesize that task-specialized teachers learn sharper, narrower modes, amplifying score-field mismatch and causing instability or collapse.

Takeaway 3: Using a task-specialized teacher as the sole guide can destabilize few-step distillation, despite stronger downstream performance.

#### 4.3 Step-wise Multi-teacher Guidance

Inspired by recent on-policy distillation methods (Xiao et al., 2026; Li et al., 2026; Fang et al., 2026), which emphasize adapting teacher supervision to the evolving student policy, we propose step-wise multi-teacher guidance. Rather than using a fixed teacher throughout training, our method constructs the DMD real-score guidance from both a stable base teacher and a set of task-specialized teachers. The contribution of each teacher is determined by the selected student distillation step and the downstream condition. Specifically, at the k-th selected student distillation step, we define the multi-teacher real-score guidance as

M

sreal(k) (xt, t, c) =

λk,m(c) s(mk)(xt, t, c), (5)

∑

m=0

where s(mk) denotes the real-score estimate from teacher Tm at the k-th selected student distillation step. The coefficient λk,m(c) ∈ [0, 1] specifies the contribution of teacher Tm under condition c, and the weights satisfy

M

∑

λk,m(c) = 1. (6)

m=0

This formulation enables the student to receive smooth and general guidance from the base teacher while selectively incorporating task-specific information from specialized teachers. Accordingly, the DMD objective with step-wise multi-teacher guidance can be written as

∇θℓDMD(k) (θ) = Eϵ,ξ,t ∇θxθ

⊺

sstu(xt, t, c) −

M

λk,m(c)s(mk)(xt, t, c) . (7)

∑

m=0

#### 4.4 Stable Distillation with Multi-teacher Guidance

In our implementation, the base teacher serves as the main anchor during the early selected distillation steps, while task-specialized teachers are incorporated selectively to provide complementary downstream guidance. Following the training protocol in Section 3, we denote the resulting distilled model as QwenImage-Flash-T2I.

Step-wise guidance stabilizes complementary supervision. Figure 3 (b) shows that the proposed step-wise multi-teacher guidance stabilizes student optimization throughout training. Unlike direct specialized-teacher distillation, which suffers from progressive quality degradation, our strategy maintains sample fidelity, layout consistency, and semantic alignment across training iterations. This indicates that the pretrained base teacher provides a stable distributional anchor, while task-specialized teachers contribute downstream-specific capabilities without inducing the severe instability observed in naive single-teacher guidance.

- Table 2: Quantitative comparison of multi-step teachers and the distilled T2I student. With only 4 NFEs, the distilled model achieves competitive performance against 80-NFE teachers, effectively inheriting their complementary strengths across landscape, portrait, and text-centric evaluation sets.

|Model<br><br>|NFEs<br><br>|Metrics|T2I-Bench Landscape Portrait Text-centric<br><br>|Average<br><br>|Rank|
|---|---|---|---|---|---|
|Qwen-Image-2.0-Base<br><br>|80<br><br>|Gemini 3.1 Pro GPT 5.5|3.52 3.64 3.08<br><br>4.24 4.30 3.73<br><br><br>|3.41 4.09<br><br>|3|
|Qwen-Image-2.0-Task-Specialized|80<br><br>|Gemini 3.1 Pro GPT 5.5|3.98 3.82 3.41<br><br>4.34 4.47 3.88<br><br><br>|3.74 4.26|1|
|Qwen-Image-Flash-T2I|4<br><br>|Gemini 3.1 Pro GPT 5.5<br><br>|3.88 3.81 3.00<br>4.30 4.41 3.75<br>|3.56 4.15<br><br>|2|

Multi-teacher guidance transfers complementary capabilities. Table 2 further verifies that step-wise multi-teacher guidance effectively transfers complementary teacher strengths to the few-step student. Although Qwen-Image-Flash-T2I uses only 4 NFEs, it achieves average scores of 3.56 with Gemini 3.1 Pro and 4.15 with GPT 5.5, surpassing the 80-NFE Qwen-Image-2.0-Base teacher in the overall ranking. These results suggest that selectively introducing specialized-teacher guidance enables the student to inherit stronger downstream capabilities while avoiding the instability caused by using a specialized teacher as the sole guidance model.

Advantages. The proposed guidance strategy has several practical advantages. First, it avoids the limitation of relying on a single teacher whose strengths may cover only part of the downstream task space. By combining a stable base teacher with task-specialized teachers, the student can preserve general generation quality while absorbing complementary expertise. Second, the method is flexible and easy to integrate into existing DMD-based distillation pipelines: any teacher with strong performance on a specific downstream task can be incorporated without changing the original objective or adding extra optimization modules. Third, the step-wise design improves training stability and reduces the need for sensitive hyperparameter tuning. Overall, step-wise multi-teacher guidance offers a lightweight, general way to transfer complementary downstream capabilities to few-step visual generators.

Takeaway 4: Step-wise multi-teacher guidance allows a few-step student to inherit complementary strengths from teachers with different downstream capabilities while maintaining stable optimization and strong generation quality.

### 5 Joint Distillation for T2I Generation and Editing

We next extend from single-task T2I distillation to joint distillation of T2I generation and instructionguided image editing into a single few-step student. This setting introduces an additional task-mixture challenge: while editing data is essential for transferring instruction-following, localized modification, content preservation, and semantic control from the teacher, excessive emphasis on editing may shift the student away from the generative distribution learned through T2I distillation. As Section 3 shows, T2I distillation is already highly sensitive to training-data composition; therefore, joint distillation must carefully balance acquiring robust editing capability with preserving the prompt following, visual fidelity, and synthesis quality of the distilled T2I model.

#### 5.1 Editing-Bench

To systematically evaluate the editing capability of few-step students, we construct Editing-Bench, a comprehensive benchmark for instruction-guided image editing. Editing-Bench contains 1,500 challenging editing cases spanning six representative categories: scene-level semantic transformation, perceptual image enhancement, object-centric manipulation, textual content editing, identity-preserving editing, and stylistic transfer. Each category includes 250 evaluation prompts, enabling a balanced assessment across both global and local editing behaviors.

Following the evaluation protocol of T2I-Bench, we use Gemini 3.1 Pro and GPT 5.5 as automatic preference-based evaluators. Higher scores indicate better instruction following, stronger source-image preservation, fewer visual artifacts, and closer alignment with human preference. Since different editing tasks emphasize different aspects of quality, we design category-specific system prompts for each evaluation split. Further details are provided in the Appendix.

- Table 3: Quantitative comparison of joint T2I-editing distillation under different T2I-to-edit data ratios on Editing-Bench. We evaluate distilled student models trained with varying mixtures of T2I and editing data across six core dimensions of instruction-guided image editing. Tea. denotes the multi-step teacher model, Qwen-Image-2.0-Task-Specialized, while Zero-shot denotes the T2I-only distilled student, Qwen-Image-Flash-T2I, evaluated directly on Editing-Bench without editing-task distillation.

|Ratio|Metrics<br><br>|Editing-Bench<br><br>Scene trans.<br><br>Perceptual enhance.<br><br>Object manip.<br><br>Text editing<br><br>Identity preserv.<br><br>Style transfer<br><br>|Average|Rank|
|---|---|---|---|---|
|Tea.<br><br>|Gemini 3.1 Pro GPT 5.5<br><br>|2.52 2.75 2.33 3.25 3.16 2.62<br>3.61 3.22 3.05 3.69 3.53 3.56<br>|2.77 3.44<br><br>|3|
|Zero-shot|Gemini 3.1 Pro GPT 5.5<br><br>|2.78 2.81 2.25 2.84 3.16 2.78<br><br>3.57 3.03 3.01 3.11 3.41 3.52<br><br><br>|2.77 3.28|4|
|9:1<br><br>|Gemini 3.1 Pro GPT 5.5|2.43 2.25 2.09 2.83 3.12 2.75<br><br>3.48 3.21 2.87 3.25 3.53 3.50<br><br><br>|2.58 3.31|5|
|7:3<br><br>|Gemini 3.1 Pro GPT 5.5<br><br>|2.49 2.80 2.34 3.35 3.18 3.05<br>3.46 2.87 3.08 3.58 3.52 3.63<br>|2.87 3.36<br><br>|2<br><br>|
|5:5<br><br>|Gemini 3.1 Pro GPT 5.5|2.86 2.25 2.68 3.18 3.19 3.68<br><br>3.66 3.06 3.20 3.13 3.47 3.92<br>|2.97 3.41<br><br>|1|

- Table 4: Quantitative analysis of T2I performance retention under different T2I-to-edit data mixtures. We evaluate jointly distilled student models trained with varying T2I-to-edit data ratios on T2I-Bench, measuring how well their T2I generation capability is preserved after incorporating editing supervision.

|Ratio|Metrics<br><br>|T2I-Bench Landscape Portrait Text-centric|Average<br><br>|Rank|
|---|---|---|---|---|
|Qwen-Image-2.0-Task-Specialized<br><br>|Gemini 3.1 Pro GPT 5.5|3.98 3.82 3.41<br><br>4.34 4.47 3.88<br><br><br>|3.74 4.26|1<br><br>|
|Qwen-Image-Flash (Zero-shot)<br><br>|Gemini 3.1 Pro GPT 5.5<br><br>|3.88 3.81 3.00<br>4.30 4.41 3.75<br>|3.56 4.15<br><br>|5|
|Qwen-Image-Flash (9:1)|Gemini 3.1 Pro GPT 5.5<br><br>|4.04 3.58 3.17 4.39 4.37 3.77|3.60 4.18<br><br>|4|
|Qwen-Image-Flash (7:3)|Gemini 3.1 Pro GPT 5.5<br><br>|3.81 3.77 3.21<br><br>4.35 4.40 3.84<br><br><br>|3.60 4.20|3|
|Qwen-Image-Flash (5:5)|Gemini 3.1 Pro GPT 5.5<br><br>|3.95 3.85 3.14<br>4.34 4.37 3.76<br>|3.65 4.16<br><br>|2|

#### 5.2 Task-mixture Composition

Given the potential tension between generation and editing supervision, we investigate how the T2I-toediting data ratio affects joint distillation. We keep the total training budget and optimization protocol fixed, and vary only the relative amount of editing data in the mixed training set. This controlled design allows us to isolate the effect of task-mixture composition and examine how different ratios influence the balance between acquiring editing capability and retaining T2I generation quality.

Concretely, we construct three joint distillation mixtures with T2I:Edit ratios of 9:1, 7:3, and 5:5. These settings cover a spectrum from T2I-dominant training to a balanced generation-editing mixture. After training, each student is evaluated on both T2I-Bench and Editing-Bench, enabling us to quantify not only how effectively editing behavior is transferred from the teacher, but also whether the original T2I capability of the distilled student is preserved or degraded as the proportion of editing data increases.

#### 5.3 Task-ratio Sensitivity in Unified Generation-Editing Distillation

Table 3 reports the Editing-Bench results under different T2I:Edit mixtures. The comparison shows that editing supervision is important for transferring instruction-guided editing behavior to the few-step student, but its effectiveness is not monotonic with the amount of editing data. Instead, joint distillation is highly sensitive to the task-mixture ratio: too little editing data provides insufficient task-specific supervision, while a more balanced mixture enables substantially stronger editing transfer.

Editing ability cannot be fully preserved by T2I-only distillation. When directly evaluated on EditingBench, the zero-shot T2I-only student, Qwen-Image-Flash-T2I, already demonstrates a certain degree of editing capability. This suggests that some instruction-guided editing behavior is retained through the underlying visual foundation model and the T2I distillation process. However, this transfer is incomplete. The zero-shot student still obtains a lower average GPT 5.5 score than the task-specialized teacher, and its performance on text editing is particularly weaker. These results indicate that T2I-only distillation is insufficient for faithfully preserving the teacher’s editing ability, especially for tasks that require precise instruction following, localized content modification, and fine-grained control over the edited region.

A balanced task mixture enables stronger editing transfer. The 9:1 mixture, which includes only a small fraction of editing data, yields the weakest performance among the jointly distilled students and even ranks below the zero-shot T2I-only baseline. This result shows that simply adding a small amount of editing data is not enough to induce reliable editing behavior. When the training distribution remains strongly dominated by T2I data, editing supervision may be too sparse to form a stable learning signal, making the student unable to consistently acquire instruction-guided editing capability.

As the editing proportion increases, Editing-Bench performance improves substantially. The 7:3 setting already outperforms both the zero-shot student and the task-specialized teacher under the Gemini 3.1 Pro average score, indicating that even a moderate amount of editing data can provide effective task-specific supervision. Among all joint distillation configurations, the balanced 5:5 mixture achieves the best overall rank and the highest average scores across both evaluators. Compared with the zero-shot T2I-only student, it raises the average score from 2.77 to 2.97 under Gemini 3.1 Pro and from 3.28 to 3.41 under GPT 5.5, while also surpassing the teacher on the former and remaining competitive on the latter. These results suggest that a balanced T2I-editing mixture provides dense and diverse editing supervision, enabling the few-step student to better inherit the teacher’s instruction-guided editing behavior. The qualitative examples in Figure 4 further support this conclusion, showing that larger editing proportions lead to more faithful instruction following, better source preservation, and higher-quality edits across representative editing categories.

Takeaway 5: The T2I:Edit ratio is a key factor in joint distillation: insufficient editing supervision can limit or even hurt editing transfer, whereas a balanced task mixture yields the strongest editing performance.

#### 5.4 Editing Supervision Benefits T2I Generation

We further examine whether introducing editing data compromises the original T2I capability of the distilled student. Surprisingly, the results in Table 4 show the opposite trend: all jointly distilled students achieve higher average T2I-Bench scores than the T2I-only distilled baseline. This indicates that editing supervision does not merely help preserve T2I generation during joint distillation; it can also provide positive transfer to the generation task itself.

One possible explanation is that instruction-guided editing introduces complementary visual-textual supervision that is not fully captured by T2I prompts alone. Editing tasks require the model to understand fine-grained instructions, localize target regions, preserve irrelevant content, and maintain consistency between the visual output and the textual command. These abilities are also beneficial for T2I generation, where strong prompt following, semantic grounding, and detailed visual-textual alignment are essential. Therefore, when properly mixed with T2I data, editing examples can serve as an auxiliary training signal that strengthens the student’s general visual-textual modeling ability rather than disrupting it.

Takeaway 6: Editing supervision provides complementary visual-textual signals, allowing joint distillation to improve T2I generation rather than simply preserving it.

### 6 Discussion

We discuss several empirical observations made during the development of Qwen-Image-Flash, including unsuccessful stabilization attempts, current limitations, and possible directions for future improvement.

#### 6.1 Unsuccessful Attempts

As shown in Section 4.2, directly using a task-specialized teacher as the real-distribution guidance can lead to structural misalignment during few-step distillation. A natural way to mitigate this issue is to

Add the text "(原王 记麻辣烫)" directly below the two "湾" characters on the left side of the signboard.

Remove the golden trumpet held in the right hand of the figure in the image.

Change the person's expression to a warm, sincere smile with bright, joyful eyes.

Change the background of the person‘s photo to a high saturation white background.

Apply the thick impasto oil painting style from Figure 1 to the line art in Figure 2.

Restore and color this photo.

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

- 5:5Tea.9:17:3Zero-shotInput

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

- Figure 4: Qualitative comparison of joint T2I-editing distillation under different task-mixture ratios. We compare editing results from the task-specialized teacher, the T2I-only zero-shot student, and jointly distilled students trained with T2I:Edit ratios of 9:1, 7:3, and 5:5 across six editing categories. The balanced
- 5:5 mixture consistently achieves better instruction following while preserving image fidelity, identity consistency, and stylistic quality, demonstrating the importance of task-ratio selection for unified few-step generation-editing distillation.

introduce an additional first-step supervision loss. Following DP-DMD (Wu et al., 2026), we experimented with adding a flow-matching objective at the first generation step, with the goal of explicitly regularizing the early structure of the generated sample.

This strategy does improve structural stability to some extent. In particular, it helps the student maintain more consistent layouts and reduces the severe geometric drift observed under direct specialized-teacher guidance. However, we also find that this benefit comes with a mild degradation in visual quality. This suggests that first-step supervision introduces a trade-off: while it constrains the student toward more stable structures, it may also restrict the distributional guidance provided by the task-specialized teacher.

#### 6.2 Limitations and Future Work

Although Qwen-Image-Flash achieves comparable T2I generation and instruction-guided editing performance to the teacher model with only 4 NFEs, several limitations remain. First, the few-step student

still struggles with highly detailed text rendering. This is especially evident in tiny text generation and complex poster-style compositions, where the model must simultaneously handle dense textual content, fine-grained typography, and precise layout control. These cases remain challenging because small errors in character shape, spacing, or text placement can significantly affect the perceived quality of the final image. Second, after incorporating editing data into joint distillation, we observe slight residual noise in some T2I outputs. This suggests that the denoising trajectory may not be fully completed in certain cases under the extremely small number of sampling steps. The issue is particularly noticeable in images with large white or clean background regions, where even subtle noise can become visually salient. Similar artifacts may also appear in recent powerful image generation systems such as GPT Image 2, indicating that this phenomenon is not unique to our model. While mild residual noise can sometimes increase perceived texture richness or naturalness in detail-heavy scenes, it is undesirable for applications that require clean backgrounds, accurate typography, and artifact-free visual layouts, such as poster generation and graphic design. We leave these limitations to future work.

### 7 Related Work

This section reviews few-step distillation and benchmarks for efficient visual generation and editing.

- 7.1 Few-step Distillation

Fast visual generation is typically enabled by distillation at the trajectory level, the distribution level, or through hybrids of the two. Trajectory-level approaches shorten the teacher’s sampling process by training a student to replace long teacher transitions with fewer update steps. Representative examples include progressive distillation (Salimans &Ho, 2022), consistency and latent consistency models (Song et al., 2023; Luo et al., 2023), as well as recent flow-based methods such as rectified flow (Liu et al., 2023a), InstaFlow (Liu et al., 2023b), and MeanFlow (Geng et al., 2025). While these methods offer strong efficiency gains, their reliance on pointwise trajectory imitation may propagate solver-induced errors and impose overly restrictive constraints on the student model. In contrast, distribution-level methods aim to match the generated distribution more directly, including adversarial diffusion distillation (Sauer et al., 2024), distribution matching distillation (DMD, Yin et al. 2024a;b), and DMD variants that separate the effects of CFG (Liu et al., 2025), enhance training stability (Bai et al., 2026), or better maintain sample diversity (Wu et al., 2026). Trajectory distribution matching connects these two paradigms by enforcing distribution-level alignment over student trajectories (Luo et al., 2026). Building on practical distribution matching, we study data composition, teacher guidance, and generation-editing task mixtures.

- 7.2 Benchmarks for Efficient Visual Generation and Editing

Evaluation protocols for T2I generation have largely been built around broad prompt collections and generic alignment metrics. Common choices include MS-COCO for FID- and CLIP-based assessment (Lin et al., 2014), together with more fine-grained benchmarks such as GenEval (Ghosh et al., 2023) and T2I-CompBench (Huang et al., 2023). These benchmarks (Ghosh et al., 2023; Huang et al., 2023; Lin et al., 2014; Wei et al., 2025), however, provide limited insight into the specific degradation patterns of modern few-step visual generators. When distillation pushes sampling to very small NFEs, models may lose accuracy in dense text rendering, structured layouts, prompt adherence, diversity, and fine visual details. Therefore, in this paper, we construct two challenging benchmarks, T2I-Bench and Editing-Bench, to systematically evaluate few-step visual generative models.

### 8 Conclusion

In this work, we revisit few-step distillation for modern visual generative foundation models through an empirical study centered on Qwen-Image-2.0, showing that effective distillation is shaped not only by objective design but also by broader training-time factors, including data composition, teacher guidance, and task mixture. Based on these findings, we develop Qwen-Image-Flash, a unified 4-NFE model capable of both high-quality T2I generation and instruction-guided image editing. More broadly, our study suggests that the next stage of efficient visual generation will not be defined solely by faster samplers or stronger losses, but by a systems-level understanding of how the entire distillation pipeline should be designed, coordinated, and scaled to fully unlock the potential of few-step visual foundation models. We hope these observations offer practical guidance for building future few-step visual foundation models that are efficient, stable, capable, and broadly applicable.

### References

Ronald T Azuma. A survey of augmented reality. Presence: teleoperators & virtual environments, 6(4): 355–385, 1997.

Lichen Bai, Zikai Zhou, Shitong Shao, Wenliang Zhong, Shuo Yang, Shuo Chen, Bojun Chen, and Zeke Xie. Optimizing few-step generation with adaptive matching distillation. arXiv preprint arXiv:2602.07345, 2026.

Tim Brooks, Aleksander Holynski, and Alexei A. Efros. InstructPix2Pix: Learning to follow image editing instructions. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 18392–18402, 2022.

Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow Transformers for highresolution image synthesis. In International Conference on Machine Learning, 2024.

Zhen Fang, Wenxuan Huang, Yu Zeng, Yiming Zhao, Shuang Chen, Kaituo Feng, Yunlong Lin, Lin Chen, Zehui Chen, Shaosheng Cao, et al. Flow-OPD: On-policy distillation for flow matching models. arXiv preprint arXiv:2605.08063, 2026.

Zhengyang Geng, Mingyang Deng, Xingjian Bai, J Zico Kolter, and Kaiming He. Mean flows for one-step generative modeling. arXiv preprint arXiv:2505.13447, 2025.

Dhruba Ghosh, Hannaneh Hajishirzi, and Ludwig Schmidt. GenEval: An object-focused framework for evaluating text-to-image alignment. In Advances in Neural Information Processing Systems, pp. 52132–52152, 2023.

Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In Advances in Neural Information Processing Systems, pp. 6840–6851, 2020.

Kaiyi Huang, Kaiyue Sun, Enze Xie, Zhenguo Li, and Xihui Liu. T2I-CompBench: A comprehensive benchmark for open-world compositional text-to-image generation. In Advances in Neural Information Processing Systems, pp. 78723–78747, 2023.

Dengyang Jiang, Dongyang Liu, Zanyi Wang, Qilong Wu, Liuzhuozheng Li, Hengzhuang Li, Xin Jin, David Liu, Changsheng Lu, Zhen Li, et al. Distribution matching distillation meets reinforcement learning. arXiv preprint arXiv:2511.13649, 2025.

Quanhao Li, Junqiu Yu, Kaixun Jiang, Yujie Wei, Zhen Xing, Pandeng Li, Ruihang Chu, Shiwei Zhang, Yu Liu, and Zuxuan Wu. DiffusionOPD: A unified perspective of on-policy distillation in diffusion models. arXiv preprint arXiv:2605.15055, 2026.

Yanyu Li, Huan Wang, Qing Jin, Ju Hu, Pavlo Chemerys, Yun Fu, Yanzhi Wang, Sergey Tulyakov, and Jian Ren. Snapfusion: Text-to-image diffusion model on mobile devices within two seconds. Advances in Neural Information Processing Systems, 36:20662–20678, 2023.

Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C Lawrence Zitnick. Microsoft COCO: Common objects in context. In European conference on computer vision, pp. 740–755, 2014.

Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. In International Conference on Machine Learning, 2022.

Dongyang Liu, Peng Gao, David Liu, Ruoyi Du, Zhen Li, Qilong Wu, Xin Jin, Sihan Cao, Shifeng Zhang, Hongsheng Li, et al. Decoupled DMD: CFG augmentation as the spear, distribution matching as the shield. arXiv preprint arXiv:2511.22677, 2025.

Jiaxiang Liu, Zhida Feng, Pengyu Zou, Zhenyu Qian, Tianrui Zhu, Jun Xia, Yuehu Dong, Yanzheng Lin, Honglin Xiong, et al. ERNIE-Image technical report. arXiv preprint arXiv:2605.25347, 2026a.

Jie Liu, Gongye Liu, Jiajun Liang, Yangguang Li, Jiaheng Liu, Xintao Wang, Pengfei Wan, Di Zhang, and Wanli Ouyang. Flow-GRPO: Training flow matching models via online RL. In Advances in neural information processing systems, pp. 40783–40818, 2026b.

Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. In International Conference on Learning Representations, 2023a.

Xingchao Liu, Xiwen Zhang, Jianzhu Ma, Jian Peng, et al. Instaflow: One step is enough for high-quality diffusion-based text-to-image generation. In International Conference on Learning Representations, 2023b.

Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In International Conference on Learning Representations, 2017.

Simian Luo, Yiqin Tan, Longbo Huang, Jian Li, and Hang Zhao. Latent consistency models: Synthesizing high-resolution images with few-step inference. arXiv preprint arXiv:2310.04378, 2023.

Yihong Luo, Tianyang Hu, Weijian Luo, and Jing Tang. TDM-R1: Reinforcing few-step diffusion models with non-differentiable reward. arXiv preprint arXiv:2603.07700, 2026.

Chaojie Mao, Chen-Wei Xie, Chongyang Zhong, Haoyou Deng, Jiaxing Zhao, Jie Xiao, Jinbo Xing, Jingfeng Zhang, Jingren Zhou, Jingyi Zhang, et al. Wan-Image: Pushing the boundaries of generative visual intelligence. arXiv preprint arXiv:2604.19858, 2026.

Chenlin Meng, Yutong He, Yang Song, Jiaming Song, Jiajun Wu, Jun-Yan Zhu, and Stefano Ermon. SDEdit: Guided image synthesis and editing with stochastic differential equations. In International Conference on Learning Representations, 2022.

Tim Salimans and Jonathan Ho. Progressive distillation for fast sampling of diffusion models. arXiv preprint arXiv:2202.00512, 2022.

Axel Sauer, Dominik Lorenz, Andreas Blattmann, and Robin Rombach. Adversarial diffusion distillation. In European Conference on Computer Vision, pp. 87–103, 2024.

Lin Song, Wenbo Li, Guoqing Ma, Wei Tang, Bo Wang, Yuan Zhang, Yijun Yang, Yicheng Xiao, Jianhui Liu, Yanbing Zhang, et al. JoyAI-Image: Awaking spatial intelligence in unified multimodal understanding and generation. arXiv preprint arXiv:2605.04128, 2026.

Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. In International Conference on Learning Representations, 2020.

Yang Song, Prafulla Dhariwal, Mark Chen, and Ilya Sutskever. Consistency models. In International Conference on Machine Learning, 2023.

Xinyu Wei, Jinrui Zhang, Zeqing Wang, Hongyang Wei, Zhen Guo, and Lei Zhang. TIIF-Bench: How does your T2I model follow your instructions? arXiv preprint arXiv:2506.02161, 2025.

Tianhe Wu, Ruibin Li, Lei Zhang, and Kede Ma. Diversity-preserved distribution matching distillation for fast visual synthesis. arXiv preprint arXiv:2602.03139, 2026.

Bangjun Xiao, Bingquan Xia, Bo Yang, Bofei Gao, Bowen Shen, Chen Zhang, Chenhong He, Chiheng Lou, Fuli Luo, Gang Wang, et al. MiMo-V2-Flash technical report. arXiv preprint arXiv:2601.02780, 2026.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.

Tianwei Yin, Michaël Gharbi, Taesung Park, Richard Zhang, Eli Shechtman, Fredo Durand, and Bill Freeman. Improved distribution matching distillation for fast image synthesis. In Advances in Neural Information Processing Systems, pp. 47455–47487, 2024a.

Tianwei Yin, Michaël Gharbi, Richard Zhang, Eli Shechtman, Fredo Durand, William T Freeman, and Taesung Park. One-step diffusion with distribution matching distillation. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 6613–6623, 2024b.

Tianwei Yin, Qiang Zhang, Richard Zhang, William T Freeman, Fredo Durand, Eli Shechtman, and Xun Huang. From slow bidirectional to fast autoregressive video diffusion models. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025.

Bing Zhao, Chenfei Wu, Deqing Li, Hao Meng, Jiahao Li, Jie Zhang, Jingren Zhou, Junyang Lin, Kaiyuan Gao, Kuan Cao, et al. Qwen-Image-2.0 technical report. arXiv preprint arXiv:2605.10730, 2026.

Yang Zhao, Yanwu Xu, Zhisheng Xiao, Haolin Jia, and Tingbo Hou. Mobilediffusion: Instant text-to-image generation on mobile devices. In European Conference on Computer Vision, pp. 225–242, 2024.

### Appendix A Evaluation Details

#### A.1 System Prompts Used in Evaluation

To ensure a reproducible, rigorous, and automated evaluation process, we employ advanced visionlanguage models (Gemini 3.0 Pro and GPT 5.5) as expert evaluators. The exact system prompts, unified templates, and category-specific rubrics used to guide the evaluator are detailed below.

T2I-Bench evaluation. For the T2I-Bench, the evaluation focuses on a dual-aspect assessment: text alignment and technical/structural quality. As shown in Table A, the system prompt instructs the evaluator to first verify whether all objects, attributes, and relationships specified in the text caption are accurately rendered. Simultaneously, it requires the model to rigorously inspect the output for granular perceptual defects, such as geometric distortion, texture melting, and bad anatomy. Objective alignment and execution are prioritized over subjective style, ensuring that a visually appealing image is still heavily penalized if it fails to follow the input prompt. To facilitate downstream parsing, the judge outputs a strict JSON object containing a holistic score and a single-sentence rationale.

Editing-Bench evaluation. Evaluating image editing tasks introduces a complex trade-off: the model must correctly execute the editing instructions while strictly preserving the unrelated regions of the source image. Due to the highly heterogeneous nature of editing operations (ranging from local text modification to global style transfer), a single static prompt is insufficient.

To resolve this, we propose a unified meta-prompt template shown in Table B. This template standardizes the basic scoring logic, the strict output JSON schema, and the penalty guidelines across all editing tasks. To adapt to different evaluation contexts, it exposes three dynamic placeholders: <category-title>, <category-rubric>, and <sub-score-criteria>.

During evaluation, the evaluation pipeline automatically detects the task category of the current sample and instantiates the meta-prompt template. Table C defines the specific values for the category titles and the corresponding focus rubrics. For instance, in Perceptual image enhancement, the rubric strictly forbids hallucinating new content, whereas in Object-centric manipulation, the rubric emphasizes plausible region filling and natural integration.

Furthermore, to prevent the evaluator from relying solely on a single impressionistic score, we enforce a multi-dimensional aspect-based scoring mechanism. Table D outlines the exact sub-score criteria keys and descriptions mapped to each category. By evaluating critical dimensions independently (such as target_localization, text_readability, or identity_preservation), the VLM judge provides granular diagnostics. Finally, as stipulated in the template, the overall_score is derived via holistic reasoning rather than a naive average, ensuring that catastrophic failures in core requirements (e.g., completely ignoring the text prompt) heavily cap the final score.

Table A: System prompt for T2I-Bench evaluation.

You are an expert evaluator for text-to-image generation results. Rate the image based on two equally critical dimensions: instruction following and visible technical quality. Do not reward an image for being merely beautiful if it fails to depict the prompt accurately.

Prompt Compliance: Verify if the image precisely reflects all objects, attributes, quantities, actions, and spatial relationships described in the caption.

Technical Quality: Focus on structural and perceptual defects: distorted geometry, warped or broken objects, inconsistent perspective, bad anatomy, face, hand, or limb deformation, texture melting, duplicated or missing parts, corrupted text or logos, blur, noise, compression artifacts, aliasing, exposure problems, color casts, unrealistic lighting, and other image-generation artifacts.

Return strict JSON only, with no markdown: {"score": <float from 1 to 5 rounded to two decimals>, "reasoning": "<one short English sentence>"}.

Table B: Unified system prompt template for Editing-Bench evaluation.

You are an expert evaluator for image editing results. You will receive one or more input/source images, the edit instruction, and one or more output image produced by an image editing model. Evaluate whether the output image correctly follows the requested edit while preserving the parts of the source image that should remain unchanged. Do not reward an image for being merely beautiful if it fails the edit. Do not penalize stylistic choices unless they conflict with the instruction or introduce visible artifacts. Use the full 1 to 5 scale.

Return strict JSON only, with no markdown and no extra text. The JSON schema is: {"overall_score": <float from 1 to 5 rounded to two decimals>, "sub_scores": {"criterion_name": <float from 1 to 5>}, "reasoning": "<one concise English sentence>", "failure_modes": ["<short phrase>"]}

Score meanings: 5 = excellent edit with correct instruction following, source preservation, natural integration, and minimal artifacts; 4 = good with minor defects; 3 = partially correct but with clear issues; 2 = mostly failed or visibly broken; 1 = unusable, ignores the instruction, or severely corrupts the image.

Category-specific rubric: <category-title> <category-rubric>

Sub-score criteria to include exactly with these keys: <sub-score-criteria>

The overall_score should be a holistic score, not a simple arithmetic average. However, severe failure in instruction following, identity preservation, or text accuracy should strongly cap the overall score.

Table C: Category-specific values for <category-title> and <category-rubric>.

<category-title> <category-rubric>

Scene-level semantic transformation

For this category, prioritize whether the whole scene transformation is semantically correct and physically coherent. A high score requires the edited scene to look like a single naturally captured image, not a pasted collage.

Perceptual image enhancement

For this category, do not reward hallucinated new content. Enhancement should improve clarity while keeping the source image faithful.

Object-centric manipulation

For deletion, the removed region should be plausibly filled. For addition/replacement, the new object must be recognizable and naturally integrated.

Textual content editing For this category, text correctness and readability are critical. A visually pleasant image with wrong or unreadable text should receive a low score.

Identity-preserving editing

For this category, identity preservation is essential. Penalize face drift, identity change, unnatural anatomy, and over-editing of unrelated regions.

Stylistic transfer For this category, style should change the appearance without destroying the source content or making the image structurally inconsistent.

#### A.2 T2I-Bench Hard Cases

T2I-Bench contains many challenging prompts that require more than simple object synthesis, which including dense text rendering, structured diagrams, multi-person interactions, fine-grained identityand complex scene layout and so on. Two random prompts of these hard samples are shown in Table E. Producing coherent images for these cases demonstrates that our model can follow detailed instructions, preserve fine visual attributes, and handle complex layouts with strong compositional control.

Table D: Category-specific values for <sub-score-criteria>.

Category <sub-score-criteria> Scene-level semantic transformation

- - Instruction following: Does the output implement the requested scene/background/composition transformation?
- - Source subject preservation: Are required people/objects/identity/clothing/details from the input preserved?
- - Global consistency: Are perspective, scale, lighting, shadows, and camera viewpoint coherent after the scene change?
- - Boundary integration: Are masks, edges, occlusion, and foregroundbackground transitions clean?
- - Visual quality: Is the final image free of distortions, texture melting, blur, noise, and generation artifacts?

Perceptual image enhancement

- - Instruction following: Does the output perform the requested enhancement, restoration, or super-resolution operation?
- - Content preservation: Does it preserve the original content, identity, geometry, colors, and layout unless the instruction asks otherwise?
- - Detail recovery: Are details sharper and cleaner without hallucinated or oversmoothed structures?
- - Artifact control: Does it avoid ringing, oversharpening, waxy texture, noise amplification, and compression artifacts?
- - Visual quality: Is the enhanced output natural, high-quality, and technically clean?

Object-centric manipulation

- - Instruction following: Is the requested object addition, deletion, replacement, or attribute modification correctly completed?
- - Target localization: Is the edit applied to the intended object/region without unintended changes elsewhere?
- - Source preservation: Are unrelated source content, identity, background, and composition preserved?
- - Physical integration: Do object scale, pose, lighting, shadows, contact, occlusion, and perspective fit the scene?
- - Visual quality: Are there no obvious seams, broken structure, duplicated parts, blur, or artifacts?

Textual content editing - Instruction following: Does the output place, replace, remove, or modify the requested text exactly as instructed?

- - Text accuracy: Are spelling, characters, capitalization, punctuation, and requested wording correct?
- - Text readability: Is the text legible, sharp, correctly oriented, and not garbled or pseudo-text?
- - Layout style match: Does the text match the original layout, font style, perspective, material, lighting, and surface geometry?
- - Source preservation: Are non-text regions and unrelated details preserved?
- - Visual quality: Is the final image free of artifacts, blur, warped letters, and unnatural overlays?

Identity-preserving editing

- - Instruction following: Does the output perform the requested clothing, pose, expression, hair, makeup, age, or face-related edit?
- - Identity preservation: Does the person keep the same recognizable identity and key facial/body characteristics where required?
- - Attribute accuracy: Are the requested changed attributes accurate and complete?
- - Anatomy and face quality: Are face, hands, limbs, pose, gaze, expression, and body proportions natural and undistorted?
- - Source preservation: Are unrelated clothing, background, accessories, and composition preserved unless the instruction changes them?
- - Visual quality: Is the final portrait technically clean, coherent, and artifact-free?

Stylistic transfer - Instruction following: Does the output apply the requested style, color tone, filter, or artistic transformation?

- - Content preservation: Are the source subject, structure, layout, identity, and important details preserved?
- - Style consistency: Is the style/tone applied consistently across the image without patchy or conflicting regions?
- - Naturalness: Are lighting, contrast, saturation, texture, and material appearance coherent after the transfer?
- - Visual quality: Is the output free of over-processing, color banding, texture collapse, blur, and artifacts?

Table E: Randomly selected hard-case prompts from T2I-Bench).

Sample Prompt

- 1 A young East Asian woman, about 20 to 28 years old, sits gracefully and casually on outdoor cement steps. She has medium-length wavy black hair with slightly curled ends falling naturally over both shoulders, a soft oval face, fair smooth skin without obvious moles or freckles, and clean natural makeup: light brown eyeshadow, thin inner eyeliner, thick curled lashes, soft pink-orange blush, matte coral-pink lips, a slight smile, and a calm cheerful gaze looking toward the upper right. She wears a white cotton camisole dress with two three-dimensional white fabric flowers on the chest and a thin drawstring tie at the neckline, plus a pale sky-blue openwork crochet cardigan with elbow-length sleeves, clear mesh texture, and a light breathable material. Her accessories include one blue-green gradient feather earring on the left ear, bright cyan-blue at the top and deep teal at the bottom, about 5 cm long, and a matching thin cord choker with a small metal clasp. Her right hand gently supports her chin, and her left hand holds an artificial bouquet placed on her lap: seven blue five-petal flowers with slightly purple petal edges, deep blue star-shaped centers, green leaves, and several white buds. The background is a traditional tie-dye craft display. In the upper-left area hangs a white sign with black text: the first larger bold line means printing and tie-dye, and the second slightly smaller line means handmade craft; the text is sharp and horizontal. Behind and to the right are several hanging tie-dye fabrics on wooden racks: a dark indigo radial spiral cloth on the left, an orange-red cloth with white spiral tie-dye near the center-right, and a dark blue cloth with white concentric-circle and radial patterns on the right, all slightly wrinkled. The ground is gray cement steps and pavement. Sunlight comes diagonally from the upper right, casting clear rectangular light patches. Use natural daylight, soft contrast, a fresh artistic handmade-craft atmosphere, a medium portrait crop from waist to mid-thigh, the subject slightly left of center with empty space on the right, a standard non-distorted lens, moderate depth of field, and a blue-white-orange color palette with a fresh healing visual style.

- 2 Create a mind map or infographic about the book The Story of Art. At the top center, place the main title The Story of Art, with the English subtitle in parentheses below it, followed by a line explaining that it is a classic narrative of art history. The main body is a relationship network centered on a portrait of E.H. Gombrich, labeled with his name. Around him are four famous artists connected by lines. In the upper left is Leonardo da Vinci, represented by the Mona Lisa, labeled with his name and described as observing nature, pursuing perfection, and being a polymathic genius; the lines between him and Gombrich are labeled analysis and interpretation, and another line points to Raphael below with a tribute label. In the lower left is Raphael, represented by an angel image, labeled with his name and described as harmonious, elegant, ideal beauty, and a master painter; his connection to Gombrich is labeled tribute and inheritance. In the upper right is Michelangelo, represented by the head of David, labeled with his name and described as ambitious, shaping power, and master sculptor; his connection to Gombrich is labeled tribute and interpretation, and another line points down to Rembrandt with a tribute label. In the lower right is Rembrandt, represented by a self-portrait, labeled with his name and described as capturing the soul, master of light and shadow, and depicting humanity; his connection to Gombrich is labeled tribute and inheritance. Across the middle of the image, draw a left-to-right wavy timeline of art history: prehistoric art with a cave-painting bull icon and a note about cave paintings and statues; ancient Egypt and Mesopotamia with pyramid and sphinx icons and a note about eternity and order; Greece and Rome with a temple icon and a note about ideal and reality; the Middle Ages with a castle icon and a note about faith and symbolism; the Renaissance with a portrait icon and a note about humanism and rebirth; Baroque and Rococo with an ornamental badge icon and a note about drama and decoration; Neoclassicism and Romanticism with a profile icon and a note about reason and emotion; and Modern Art with an atom icon and a note about experimentation and abstraction. Below the timeline, add a centered sentence saying that art is not static but a constantly developing and changing story. At the bottom left, create a Core Themes section with five icons and labels: art-history primer, visual literacy, classic work, insight, and human spirit. At the bottom right, create a Character Cards section with two dark rectangular cards: one for Leonardo da Vinci with a side-profile silhouette and traits of curiosity, broad learning, and perfectionism, and one for Michelangelo with a side-profile silhouette and traits of determination, passion, and great ambition.

