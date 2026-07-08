# arXiv:2601.13976v2[cs.CV]23Jan2026

FantasyVLN: Unified Multimodal Chain-ofThought Reasoning for Vision-Language Navigation

Jing Zuo1,2*§, Lingzhou Mu1,3*§, Fan Jiang1*†‡, Chengcheng Ma1, Mu Xu1, Yonggang Qi2‡

1Fantasy AIGC Team, 2Beijing University of Posts and Telecommunications, 3Tsinghua University

jiangfan0576@gmail.com qiyg@bupt.edu.cn

Abstract

Achieving human-level performance in Vision-and-Language Navigation (VLN) requires an embodied agent to jointly understand multimodal instructions and visual-spatial context while reasoning over long action sequences. Recent works, such as NavCoT and NavGPT-2, demonstrate the potential of Chain-of-Thought (CoT) reasoning for improving interpretability and long-horizon planning. Moreover, multimodal extensions like OctoNav-R1 and CoT-VLA further validate CoT as a promising pathway toward human-like navigation reasoning. However, existing approaches face critical drawbacks: purely textual CoTs lack spatial grounding and easily overfit to sparse annotated reasoning steps, while multimodal CoTs incur severe token inflation by generating imagined visual observations, making real-time navigation impractical. In this work, we propose FantasyVLN, a unified implicit reasoning framework that preserves the benefits of CoT reasoning without explicit token overhead. Specifically, imagined visual tokens are encoded into a compact latent space using a pretrained Visual AutoRegressor (VAR) during CoT reasoning training, and the model jointly learns from textual, visual, and multimodal CoT modes under a unified multiCoT strategy. At inference, our model performs direct instruction-to-action mapping while still enjoying reasoning-aware representations. Extensive experiments on LH-VLN show that our approach achieves reasoning-aware yet real-time navigation, improving success rates and efficiency while reducing inference latency by an order of magnitude compared to explicit CoT methods. Project: https://fantasy-amap.github.io/fantasy-vln/

1 Introduction

Vision-and-LanguageNavigation(VLN)aimstoenableanembodiedagenttofollownatural-language instructions and navigate complex visual environments Krantz et al. (2020); Wu et al. (2024); Anderson et al. (2018); Gu et al. (2022). Solving this task requires the joint understanding of semantics from language and spatial geometry from visual observations, along with long-horizon reasoning to plan a sequence of actions. In particular, for multi-stage and long-horizon navigation scenarios as proposed in (Song et al., 2025), the ability to perform robust multimodal reasoning, i.e., to integrate linguistic intent with visual-spatial context over extended temporal dependencies, is especially critical. Despite the progress made by recent multimodal large models, achieving effective

∗Equal contribution. †Project leader. ‡Corresponding author. §Work done during internship at Fantasy AIGC Team.

[Figure 2]

[Figure 3]

###### Fantasy AIGC Team

|Legend<br><br>Predicted action via T-CoT Predicted action w/o CoT Predicted action via V-CoT<br><br>Cross-mode alignment constraint|
|---|

[Figure 4]

Semantic Planning: To ﬁnish this task, the agent should ﬁrst ﬁnd ...

###### Visual Description: The historical

observations indicate that ... The current visual inputs show that ...

Action Planning: Move forward three times to exit the bathroom, ...

Subgoal 2: Go to the bathroom.

Visual Imagination: After executing the planned actions, ... I should see...

Subgoal 3: Go ﬁnd the toilet in the bathroom.

Semantic Plans

###### 🏁 Finish

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

Subgoal 1: Find the toilet paper.

🚩 Global Task: Bring the toilet paper from the bedroom to the toilet in the bathroom.

Imagined Observation

- Figure 1: Overview of FantasyVLN. FantasyVLN is a VLNframeworkthatintegrates the strengths of textual and visual CoT reasoning modes, thereby jointly modeling semantic planning and spatial understanding.

cross-modal reasoning in VLN remains challenging due to the semantic–spatial gap and the need for interpretable yet sample-efficient reasoning mechanisms.

The recent success of large language models (LLMs) has inspired the integration of Chain-ofThought (CoT) reasoning into embodied navigation to improve interpretability and long-horizon decision-making. Methods such as NavCoT (Lin et al., 2025b) and NavGPT-2 (Zhou et al., 2024) employ step-by-step textual reasoning to decompose navigation instructions or generate intermediate subgoals. However, their reasoning remains confined to the textual modality, typically by translating observations into captions, thereby limiting the joint modeling of semantic planning and spatial understanding, both essential for successful navigation. This limitation is compounded by the difficulty of annotating CoT supervision in VLN, as highlighted by EvolveNav (Lin et al., 2025a), where multiple valid action sequences often exist. Moreover, explicitly supervised CoT reasoning tends to overfit training distributions and generalize poorly to unseen environments.

Lately, works such as CoT-VLA (Zhao et al., 2025), VISTA (Huang et al., 2025), RBF++ (Chen et al., 2025), OctoNav-R1 (Gao et al., 2025), and OmniNav (Xue et al., 2025) have extended CoT reasoning into visual or multimodal domains to better couple semantic and spatial reasoning for generalizability. While this multimodal CoT paradigm marks an important step forward, it also introduces new challenges for long-horizon navigation. In particular, modeling reasoning chains across both language and vision requires the model to iteratively generate and interpret imagined intermediate observations at each step, leading to severe token inflation. A typical reasoning step spanning 5–7 actions expands into over 3k–5k tokens, an order of magnitude larger than purely textual CoTs (usually <500 tokens). This explosion in sequence length drastically increases both training and inference latency, rendering real-time navigation infeasible even on high-end GPUs.

To address these challenges, we propose a unified implicit reasoning framework that retains the benefits of CoT-style reasoning while eliminating its explicit token overhead during inference. The key idea is twofold: (i) During training, we encode the imagined observation tokens generated by multimodal CoT reasoning into a compact latent space using a pretrained Visual AutoRegressive (VAR) model. This significantly reduces sequence length and training cost without compromising the richness of visual reasoning. (ii) At inference, the agent performs direct instruction-to-action mapping while still leveraging reasoning-aware representations, inspired by the train-with-CoT, infer-without-CoT paradigm of Aux-Think (Wang et al., 2025).

Concretely, we introduce a unified multi-CoT training strategy that jointly learns from textual-only, visual-only, and textual–visual CoT modes using a special tag token to indicate each mode. This design unifies both the input format and model parameters within a single framework. During training, we align the action predictions from CoT-based reasoning modes with those from direct

prediction (without CoT), enforcing modality-invariant reasoning representations. Consequently, the model learns implicit reasoning capabilities that generalize effectively without explicit CoT supervision or overfitting to training distributions.

To this end, our contributions are summarized as follows: (i) We propose the first unified implicit CoT reasoning framework that integrates textual, visual, and multimodal CoT paradigms within a single model. Unlike prior explicit CoT methods, our approach trains with diverse reasoning modes but performs inference without generating CoT sequences, achieving reasoning-aware yet real-time navigation. (ii) We introduce a gating-based multi-CoT learning mechanism that allows seamless switching among reasoning modes and direct action prediction under shared parameters. By aligning CoT-driven and direct action predictions, our model learns consistent, modality-invariant reasoning representations. (iii) To reduce the token explosion in multimodal reasoning, we compress imagined observation tokens into a compact latent space using a pretrained Visual AutoRegressor (VAR), improving training efficiency while preserving semantic–spatial reasoning capacity. (iv) Extensive experiments on the challenging LH-VLN benchmark demonstrate that our method substantially improves navigation success and efficiency in multi-stage and long-horizon scenarios, while reducing inference latency by an order of magnitude compared to explicit CoT approaches.

Cross-Mode Alignment Loss

|Legend<br><br>Textual Token Visual Token Action Token Gating Token Model Input AR Feedback VAR Decode<br><br>|
|---|

VAR Decode VAR Decode

Imagined Observation

###### Imagined Observation Semantic Plans

Semantic Plans

[Figure 10]

Visual Description: The historical images shows... The current images shows a living room with a door on my right...

[Figure 11]

Visual Description: The historical images shows...

T-CoT Action

V-CoT Action

T-COT V-CoT Action

Action

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

Reasoning: ... A right turn could logically lead to...

Reasoning: ... A right turn could logically lead to the ofﬁce based on typical house layouts...

###### Visual Imagination:

Visual Imagination: ... After turning right, a doorway leading to the ofﬁce should become visible, with ofﬁce-

... After turning right, a doorway leading to...

FantasyVLN

0 0 1 0 0 1 1 1

Instruction Observation

Instruction Observation

Instruction Observation

Instruction Observation

(b) Textual CoT Reasoning (Training Only)

(a) Non-CoT Reasoning

(c) Visual CoT Reasoning (Training Only)

(d) Multimodal CoT Reasoning (Training Only)

- Figure 2: Overview of our unified multimodal Chain-of-Thought reasoning framework. The model supports four reasoning modes under a shared architecture: (a) non-CoT reasoning for real-time inference, (b) textual CoT, (c) visual CoT enabled by VAR-compressed imagined observations, and (d) multimodal CoT combining textual and visual reasoning. A gating mechanism switches the model across reasoning modes, while the action predictions from CoT modes are consistently aligned with the non-CoT mode.

2 Related Works

- 2.1 Vision-and-Language Navigation

Early VLN models typically separate perception, instruction understanding, and action planning into discrete modules, and rely on imitation (Nguyen et al., 2019; Wang et al., 2022; Wu et al., 2020) or reinforcement learning (Xu et al., 2023; Wang et al., 2020) with auxiliary tasks such as progress monitoring or instruction reweighting. However, these methods, built on panoramic observations in discrete environments (e.g., R2R (Anderson et al., 2018) and RxR (Ku et al., 2020)), suffer from poor semantic alignment and limited generalization in continuous or unseen environments (e.g. VLN-CE (Krantz et al., 2020)). To address these limitations, recent studies have shifted toward end-to-end navigation policy learning with pretrained vision-language models. For example, Poliformer (Zeng et al., 2025) introduces transformer-based on-policy reinforcement learning for video-level navigation. NaVid (Zhang et al., 2024a) and Uni-NaVid (Zhang et al., 2025a) extend this paradigm by performing monocular video-based navigation without depth or maps and unifying multiple embodied tasks. NaVILA (Cheng et al., 2024) further integrates VLN with legged robot locomotion, achieving impressive cross-embodiment generalization. While achieving remarkable

progress on short-term tasks, they still struggle to reason and plan for long-horizon, multi-stage tasks. More recently, CoT reasoning has emerged as a crucial paradigm for Embodied AI tasks. In VLN, NavGPT leverages the zero-shot CoT reasoning ability of GPT-4, while Aux-Think introduces auxiliary CoT supervision to internalize reasoning patterns during training. Yet, existing CoTbased VLN methods confine reasoning within a single modality, leaving multimodal CoT largely unexplored. Inthispaper, wefollowthecontinuousenvironmentsettingandsystematicallyinvestigate multimodal CoT reasoning in VLN.

- 2.2 Chain-of-Thought Reasoning

Chain-of-Thought(CoT)reasoningenableslargelanguagemodels(LLMs)tosolvecomplexproblems by explicitly generating intermediate steps (Wei et al., 2022). Subsequent variants, such as SelfConsistency (Wang et al., 2023) and Least-to-Most Prompting (Zhou et al., 2023), further enhance reasoning robustness and compositionality. Recent studies have extended CoT to vision-language models (Zhang et al., 2024b), which can be categorized into three types based on the modality of reasoning steps: Textual CoT, Visual CoT, and Multimodal CoT. Specifically, Textual CoT (Zhang

- et al., 2024c) in VLMs typically follows the format of vanilla LLM CoT. Visual CoT methods, such as CoT-VLA (Zhao et al., 2025) and DreamVLA (Zhang et al., 2025b), generate future frames before action prediction in manipulation tasks, while Multimodal CoT (Cheng et al., 2025) jointly predicts paired textual and visual reasoning steps in multimodal tasks. To the best of our knowledge, FantasyVLN is the first unified CoT reasoning framework that integrates these three reasoning paradigms.

- 3 Methods

- 3.1 Overview

We propose FantasyVLN, a VLN framework that integrates multimodal reasoning modes as its core design, while enabling implicit reasoning for efficient inference. As shown in Figure 2, FantasyVLN internalizes diverse CoT reasoning patterns across modalities through end-to-end joint training, and enhances the non-CoT reasoning mode via a cross-mode alignment constraint. This enables combining the advantages of both textual and visual CoT reasoning without incurring explicit CoT reasoning time. Moreover, we perform visual CoT reasoning in the latent space of the VAR model (Tian et al., 2024), which improves training and inference efficiency compared with pixel-space methods.

Below, weintroducetheproblemsetup, cross-modealignmentconstraint, unifiedmultimodalimplicit reasoning and latent visual CoT learning.

- 3.2 Problem Setup

VLN aims to develop an embodied agent πθ that navigates continuous 3D environments O based on a natural language instruction I and visual observations, which can be formulated as a non-Markovian

temporal decision problem. Let s0 denote the initial state, i.e., location and orientation, and U denote the action space. At each timestep t, the agent πθ receives multi-view visual observations ot ∈ O and predicts future actions At ∈ U conditioned on the instruction I and historical observations {o≤t}. Subsequently, the predicted actions At are executed, transferring the agent πθ to a new state according to the environment dynamics. This interaction process continues until a stop action is executed or the maximum step T is reached.

- 3.3 Compact Visual Chain-of-Thought

Conventional V-CoT reasoning predicts thousands of visual tokens at each reasoning step, resulting in low training efficiency and high inference latency. To address this issue, we present Compact Visual Chain-of-Thought (CompV-CoT), which trains Qwen2.5-VL to directly generate a compact set of visual tokens in the latent space of a pretrained VAR model, yielding a novel compressed visual chain-of-thought representation with far fewer tokens. The VAR model follows a next-scale prediction paradigm to hierarchically encode visual information, achieving higher efficiency than

conventional autoencoding approaches such as VAE, VQ-VAE (Gafni et al., 2022) or RAE (Zheng

- et al., 2025). Given a 256 × 256 image, the VAR model enables precise reconstruction using the corresponding low-scale representations, which contain only 30 visual tokens. As shown in Table 1, VAR achieves a higher compression ratio under comparable reconstruction quality.

- Table 1: Comparison of compression ratio and reconstruction error (MSE) across different visual compressors.

###### Compressors Comp. Ratio MSE

RAE-DINOv2-B 1/256 0.012 RAE-SigLIP2-B 1/256 0.011

VAE 1/64 0.005 VQVAE 1/64 0.007

VAR 1/2185 0.039

Specifically, we employ the VAR model as the visual decoder of the VLM and perform V-CoT in the VAR latent space. The VLM first takes navigation instructions and visual observations as input and then generates latent representations of future observations before predicting actions. The VAR model finally decodes generated representations into pixel frames. We freeze the VAR model during training, while the VLM first learns to predict latent future observations and then infers the corresponding actions. During inference, we use only the VLM to perform visual CoTbased navigation without explicit VAR decoding. Owing to the highly efficient visual information compression and non-display image decoding, the proposed CompV-CoT method improves both training and inference efficiency.

- 3.4 Unified Multimodal Chain-of-Thought

Building on CompV-CoT, we further present a Unified Multimodal Chain-of-Thought (UM-CoT) framework that integrates textual, compressed visual, and multimodal reasoning within a single agent.

Textual CoT in VLN. Textual CoT (T-CoT) models the agent’s reasoning as an explicit semantic planning process that bridges language understanding and action decision. Instead of directly

mapping instructions to actions, the agent first generates textual intermediate reasoning steps Tt. These reasoning steps then provide structured causal guidance for predicting subsequent actions

At, enabling interpretable and more reliable decision-making. Specifically, the intermediate steps typically involve inferring subgoals from the instruction, assessing progress through current and historical visual observations, and identifying actionable cues for achieving the next objective.

CompV-CoT as Visual CoT. For visual CoT (V-CoT), we directly adopt the CompV-CoT introduced in Sec. 3.3 as the visual reasoning mode in UM-CoT. In this setting, the agent imagines future observations in the VAR latent space by predicting compressed visual tokens, and then infers actions conditioned on the imagined latent trajectory. Compared with pixel-space prediction, this CompV-CoT design yields more efficient and stable visual reasoning.

Multimodal CoT in VLN. Multimodal CoT (MM-CoT) is defined as a native combination of TCoT and CompV-CoT, where the agent is required to generate paired textual–visual reasoning steps. We denote the multimodal reasoning trace as Mt = [Tt,Vt], which jointly encodes semantic plans and imagined future observations, and use it to guide subsequent action prediction.

Unified Multimodal CoT Reasoning Framework. To unify the above reasoning modes within a single framework, we introduce two binary gating signals gT and gV that control whether textual and visual reasoning is activated. Given I, {o≤t}, and (gT ,gV), the agent jointly predicts reasoning traces and actions:

[ Rt, At] = πθ I,{o≤t},gT ,gV , (1)

where

 

None, if (gT ,gV) = (0,0), Tt, if (gT ,gV) = (1,0), Vt, if (gT ,gV) = (0,1), Mt, if (gT ,gV) = (1,1).

(2)

Rt =



This gating mechanism allows a single policy to flexibly operate in CoT, T-CoT, CompV-CoT, and MM-CoT modes.

Joint Training via Data Mixture. Given the navigation instruction I, visual observations {o≤t} and the ground truth action At. To enable end-to-end training, we organize the expert navigation dataset D into five-tuples:

###### [I,{o≤t},Tt,Vt,At] ∈ D, (3)

where Tt and Vt denote the ground truth textual reasoning steps and CompV-CoT visual reasoning steps, respectively. We employ Qwen-VL-Max to generate textual reasoning traces Tt. During training, (gT ,gV) are uniformly sampled and integrated with I and {o≤t} to form the query, while the answer is constructed according to Eq. (2) by selecting Rt ∈ {None,Tt,Vt,Mt} together with At. The joint objective is defined as:

LJoint = (¬gT ∧ ¬gV)LCE At,At

+ (gT ∧ ¬gV)LCE [ Tt, At],[Tt,At] + (¬gT ∧ gV)LCE [ Vt, At],[Vt,At] + (gT ∧ gV)LCE [ Mt, At],[Mt,At] ,

(4)

where LCE denotes the causal cross-entropy loss. Algorithm 1 Cross-Mode Aligned Joint Training

- 1: Input: Dataset D, parameters θ, learning rate η, alignment weight λalign
- 2: Output: Trained parameters θ∗
- 3: while not converged do
- 4: [I,{o≤t},Tt,Vt,At] ∼ D
- 5: At ← πθ(I,{o≤t},gT =0,gV=0)
- 6: θ ← θ − η∇θLCE( At,At)
- 7: At ← sg πθ(I,{o≤t},gT =0,gV=0)
- 8: [ Tt, ATt ] ← πθ(I,{o≤t},gT =1,gV=0)
- 9: [ Vt, AVt ] ← πθ(I,{o≤t},gT =0,gV=1)
- 10: [ Mt, AMt ] ← πθ(I,{o≤t},gT =1,gV=1)
- 11: Compute L∗Joint using Eq. (7)
- 12: θ ← θ − η∇θL∗Joint
- 13: end while
- 14: θ∗ ← θ
- 15: return θ∗

- 3.5 Cross-Mode Alignment Constraint

To prevant conflict between different reasoning modes, we introduce a Cross-Mode Alignment Constraint that regularizes the unified multimodal CoT training. The key idea is to use the non-CoT reasoningmodeasasupervisorysignaltoalignallCoTvariants, therebyembeddingdiversereasoning

behaviors into a shared latent policy. Let At, ATt , AVt , and AMt denote the action predictions from the non-CoT, T-CoT, V-CoT, and MM-CoT reasoning modes, respectively. In each iteration, we first optimize the non-CoT reasoning mode with the objective:

###### Lnon-CoT = LCE At,At , (5)

- Table 2: Comparison of navigation accuracy across different VLN methods on LH-VLN benchmark. The best and second-best results are marked in bold and underlined, respectively.

CoT Modal Methods SR ISR CSR CGT

None/ZS

Random 0 0 0 0 GLM-4v prompt 0 0 0 0 GPT-4 + NaviLLM 0 2.19 1.45 2.61

MGDM 0 2.34 1.65 2.91

Visual

CoT-VLA 0 0 0 0 WorldVLA 0 0 0 0

Textual Aux-Think 0.65 3.16 2.04 1.47 unified multimodal FantasyVLN 2.44 11.01 9.64 8.99

where

At = πθ I,{o≤t},gT = 0,gV = 0 . (6)

We then obtain the soft targets At by rerunning the forward process (6). Finally, we incorporate the cross-mode alignment constraint into the joint objective of unified multimodal CoT reasoning:

L∗Joint = LAlign + LCoT, (7) where

LAlign = LCE ATt , At

+ LCE AVt , At + LCE AMt , At ,

(8) and

LCoT = LCE Tt,Tt

+ LCE Vt,Vt + LCE Mt,Mt .

(9)

We alternately minimize the non-CoT objective (5) and the cross-mode aligned joint objective (7) until the training losses Lnon-CoT and L∗Joint converge. During this alternating optimization, all reasoning modes operate on similar inputs, share network parameters, and are aligned to identical supervisory signals, thereby implicitly embedding diverse CoT reasoning patterns into a unified latent representation. The overall algorithm is presented in Algorithm 1.

3.6 VLN During Inference

Due to the real-time demands of VLN and the inference latency introduced by explicit CoT token decoding, we follow Aux-Think (Wang et al., 2025) and adopt the non-CoT reasoning mode during inference. Similar to Aux-Think (Wang et al., 2025), our framework serves as an implicit reasoning mechanismthatinternalizesdiverseCoTpatternsandimplicitlyenhancesnon-CoTreasoningthrough cross-mode alignment and joint training across reasoning modes.

4 Experiments

- 4.1 Experimental Setup

Benchmark. We evaluate FantasyVLN on the challenging LH-VLN (Song et al., 2025) benchmark, which is characterized by multi-stage tasks and long navigation trajectories. On one hand,

- Table 3: Comparison of navigation accuracy across different reasoning mode combinations on LHVLN.

non-CoT T-CoT V-CoT MM-CoT SR ISR CSR CGT ✓ 0 2.01 1.51 1.55

- ✓ ✓ 0.98 8.26 6.60 6.15

- ✓ ✓ 1.46 11.19 9.66 8.84

✓ ✓ 0.49 7.77 6.48 8.89 ✓ ✓ ✓ ✓ 2.44 11.01 9.64 8.99

- Table 4: Comparison of inference efficiency across different CoT reasoning methods. The best results are marked in bold.

Reasoning Mode Methods Model Size APS Explicit CoT-VLA 7B 0.19

WorldVLA 7B 1.02 Aux-Think 8B 0.97

Implicit

FantasyVLN 7B 1.03

multi-stage navigation requires the agent to sequentially reach multiple goals, imposing higher demands on reasoning and planning. On the other hand, longer navigation trajectories amplify cumulative errors compared to shorter ones. Following the standard LH-VLN setting, we perform online evaluation on the test set, where both the tasks and scenes are unseen.

Baselines. We compared the proposed FantasyVLN with several representative methods. They can be divided into four categories: (i) textual CoT-based methods, Aux-Think (Wang et al., 2025); (ii) visual CoT-based methods, CoT-VLA and WorldVLA; (iii) memory-based methods, MGDM; (iv) other baselines provided by LH-VLN, GLM-4v prompt, NaviLLM and GPT-4 + NaviLLM. For fair comparison, all methods are trained on the same LH-VLN training set, and the validation set is used to select the best checkpoint for each method. For Aux-Think and CoT-VLA, we implement their methods based on the descriptions in their papers, as the training codes are not publicly available. For WorldVLA, we adapt the official implementation by modifying the preprocessing pipeline to support training on the LH-VLN dataset. For all other methods, we use the implementations provided by LH-VLN.

Metrics. Following (Song et al., 2025), we use Success Rate (SR), Independent Success Rate (ISR), Conditional Success Rate (CSR), and CSR weighted by Ground Truth (CGT) to measure multi-stage navigation accuracy. SR denotes the success rate of multi-stage task navigation, ISR represents the success rate of individual subtasks, CSR weights the ISR according to the success of preceding subtasks, and CGT further weights the CSR based on the length of the expert trajectory. Moreover, we introduce the Action Per Second (APS) to evaluate inference efficiency:

Nact Tnav

APS =

, (10)

where Nact denotes the total number of executed actions, and Tnav represents the total navigation time in seconds.

- 4.2 Main Results

Navigation Accuracy. Table 2 presents the quantitative results of navigation accuracy across different VLN methods on the LH-VLN benchmark. FantasyVLN achieves superior performance across all metrics, with SR, ISR, CSR, and CGT of 2.44, 11.01, 9.64, and 8.99, respectively, significantly surpassing all baselines. Aux-Think shows suboptimal results in SR, ISR, and CSR, indicating that T-CoT enhances navigation robustness compared to non-CoT approaches. However, its performance still exhibits a notable gap compared to FantasyVLN, owing to the limitations of single-modal CoT modeling and the lack of an explicit–implicit alignment mechanism. MGDM performs relatively well among non-CoT baselines, particularly in CGT, suggesting that memory mechanisms offer limited yet tangible benefits. Overall, the results demonstrate that our unified multimodal implicit reasoning framework is crucial for tackling the complex multi-stage VLN task.

Inference Efficiency. To quantify the inference efficiency of different CoT reasoning methods, we report APS in Table 4. Implicit reasoning models, including FantasyVLN, Aux-Think, and WorldVLA, exhibit comparable efficiency and outperform the explicit approach CoT-VLA by a substantial margin. This outcome is expected. Implicit reasoning predicts each action by decoding a single token, while explicit reasoning requires generating CoT reasoning steps with thousands of tokens. Under similar model sizes, implicit CoT reasoning predicts approximately one action

per second, while explicit CoT reasoning yields only 0.19 actions per second. Therefore, implicit reasoning better satisfies the real-time requirements of the VLN task.

- 4.3 Ablation Studies

Contribution of Each Reasoning Mode. FantasyVLN integrates diverse reasoning modes within a unified framework. To verify the contribution of each reasoning mode to the overall framework, we explore various combinations of non-CoT, T-CoT, V-CoT, and MM-CoT modes during training. As shown in Table 3, combining any CoT reasoning mode with non-CoT reasoning consistently improves navigation performance across all metrics. Integrating all four reasoning modes further enhances the overall performance.

VAR Scale Selection. As detailed in Section 3.3, we perform V-CoT in the latent space of VAR. To select the optimal VAR scale for latent V-CoT learning, we conduct comprehensive ablation studies on a subset of LH-VLN. We first report the ISR results across different VAR scales, ranging from 1 to 10, as shown in Figure 3. The results show that scale 4 achieves the best performance. We attribute this to smaller scales lacking sufficient visual information, while larger scales leading redundancy. To validate this argument, we randomly sample 100 images from LH-VLN and employ a pretrained VAR model to reconstruct them. Specifically, VAR takes the ground truth latents up to a given scale as input and predicts the remaining latent scales. The reconstructed images are then obtained by decoding both the ground truth and predicted latents together. As shown in Figure 4, the results are consistent with our argument.

Effect of Cross-Mode Alignment Constraint. We introduce a cross-mode alignment constraint into the joint training of FantasyVLN. Table 5 compares SR, ISR, CSR, and CGT performance with and without this constraint. Without the cross-mode alignment constraint (✗), FantasyVLN exhibits weak navigation ability, achieving only marginal ISR, CSR, and CGT scores. Adopting this constraint (✓) yields substantial improvements across all metrics, with SR increasing from 0 to 2.44, ISR from 2.39 to 11.01, CSR from 1.19 to 9.64, and CGT from 1.28 to 8.99. This indicates that cross-mode alignment is essential for FantasyVLN, a unified framework that integrates diverse reasoning modes.

8

7.41

6.00

6

### ISR

3.76

3.76 3.76

4

2.26

2

0.75 0.90

0.00 0.00

0

0 1 2 3 4 5 6 7 8 9

### VAR Scale

Figure 3: ISR variation with respect to different VAR scales.

###### GT Scale 1 Scale 2 Scale 3 Scale 4 Scale 5 Scale 6 Scale 7 Scale 8 Scale 9 Scale 10

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

Figure 4: Qualitative comparison of image reconstruction results produced by the VAR model using latent inputs across different scales. For each image, the VAR model receives the ground truth latents up to a specified scale and predicts all remaining scales; the final reconstruction is obtained by decoding the combined ground truth and predicted latents.

- Table 5: Comparison of SR, ISR, CSR, and CGT performance with and without cross-mode alignment.

Alignment Constraint SR ISR CSR CGT

✗ 0 2.39 1.19 1.28 ✓ 2.44 11.01 9.64 8.99

- 4.4 More Results

Training Efficiency. As shown in Table 2, existing visual CoT methods (e.g., WorldVLA) achieve limited navigation accuracy on the LH-VLN benchmark, failing to generalize to long-horizon scenarios. To understand the underlying cause, we compare their training efficiency with our unified multimodal formulation. As shown in Figure 5, WorldVLA exhibits slow and unstable convergence, requiring over 10k iterations to reach moderate token prediction accuracy. This indicates that pixellevel V-CoT learning delivers weak gradient signals, as the model must reconstruct high-dimensional visual tokens for each reasoning step. In contrast, FantasyVLN converges rapidly within a few thousand iterations, reflecting stable supervision and more efficient learning dynamics. This improvement stems from our CompV-CoT design, where visual reasoning operates in a compact latent space encoded by the pretrained VAR compressor. By replacing dense pixel reconstruction with compressed latent prediction, the model learns richer multimodal reasoning cues under a substantially lighter optimization burden. Overall, these results highlight that CompV-CoT not only enhances reasoning efficiency but also yields more stable and interpretable learning behavior, contributing to the superior navigation accuracy of FantasyVLN in long-horizon tasks.

Explicit vs. Implicit Reasoning. FantasyVLN supports both explicit and implicit reasoning, enabling us to systematically compare their effectiveness across different CoT modalities. As summarized in Table 6, we evaluate the T-CoT, V-CoT, and MM-CoT models under two inference modes. For each case, the model is jointly trained with its corresponding CoT mode and the direct prediction pathway. During inference, using the CoT branch corresponds to explicit reasoning, while employing the direct pathway corresponds to implicit reasoning. Overall, implicit reasoning consistently yields higher navigation accuracy, particularly under multimodal settings. In MM-CoT, implicit inference achieves the best performance with 2.44 SR and 11.01 ISR, surpassing the explicit counterpart by a large margin. This result aligns with the observations in Aux-Think (Wang et al., 2025), suggesting that explicit CoT decoding may amplify cumulative reasoning errors across long trajectories. We attribute this phenomenon to two key factors: (i) the limited training data of LH-VLN (only 18k trajectory slices of five steps each) makes explicit CoT sequences prone to overfitting and error propagation; (ii) explicit reasoning expands temporal dependencies, causing misaligned textual or visual CoT tokens to accumulate deviations over time. In contrast, implicit reasoning benefits from crossmode alignment during training, allowing the model to internalize reasoning cues while maintaining stable and efficient inference.

1.0

0.8

##### TrainTokenAccuracy

0.6

0.4

0.2

WorldVLA

FantasyVLN

0.0

0 2000 4000 6000 8000 10000 12000 14000

Iterations

Figure 5: Comparison of training efficiency between FantasyVLN and WorldVLA.

Table 6: Comparison of explicit and implicit CoT reasoning across modalities.

Metrics Mode T-CoT V-CoT MM-CoT SR

explicit 0.98 0.49 0.98 implicit 0.49 1.46 2.44

explicit 8.26 7.34 8.62

ISR

implicit 6.06 11.19 11.01

#### 5 Conclusion

We introduced FantasyVLN, a unified implicit reasoning framework that preserves the benefits of Chain-of-Thought supervision while avoiding the token explosion inherent to explicit textual or multimodal CoTs. By compressing imagined visual observations into a compact latent space via a pretrained VAR model and jointly training across textual, visual, and multimodal CoT modes under a unified multi-CoT strategy, the framework learns modality-invariant reasoning representations without requiring explicit CoT generation at inference. As a result, the agent performs direct instruction-to-action mapping while retaining reasoning-aware behavior. Experiments on the challenging LH-VLN benchmark show that this formulation substantially improves navigation accuracy and efficiency, while reducing inference latency by an order of magnitude compared to explicit CoT baselines. These findings demonstrate that implicit multimodal reasoning provides a practical pathway toward real-time embodied navigation, and highlight the potential of compact latent reasoning signals for closing the gap between semantic intent and spatial decision-making in complex environments.

#### References

Peter Anderson, Qi Wu, Damien Teney, Jake Bruce, Mark Johnson, Niko S¨underhauf, Ian Reid, Stephen Gould, and Anton Van Den Hengel. Vision-and-language navigation: Interpreting visually-grounded navigation

instructions in real environments. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 3674–3683, 2018.

Qiguang Chen, Libo Qin, Jinhao Liu, Yue Liao, Jiaqi Wang, Jingxuan Zhou, and Wanxiang Che. Rbf++: Quantifying and optimizing reasoning boundaries across measurable and unmeasurable capabilities for chain-of-thought reasoning. arXiv preprint arXiv:2505.13307, 2025.

An-Chieh Cheng, Yandong Ji, Zhaojing Yang, Zaitian Gongye, Xueyan Zou, Jan Kautz, Erdem Bıyık, Hongxu Yin, Sifei Liu, and Xiaolong Wang. Navila: Legged robot vision-language-action model for navigation. arXiv preprint arXiv:2412.04453, 2024.

Zihui Cheng, Qiguang Chen, Xiao Xu, Jiaqi Wang, Weiyun Wang, Hao Fei, Yidong Wang, Alex Jinpeng Wang, Zhi Chen, Wanxiang Che, et al. Visual thoughts: A unified perspective of understanding multimodal chain-of-thought. arXiv preprint arXiv:2505.15510, 2025.

Oran Gafni, Adam Polyak, Oron Ashual, Shelly Sheynin, Devi Parikh, and Yaniv Taigman. Make-a-scene: Scene-based text-to-image generation with human priors. In European conference on computer vision, pp. 89–106. Springer, 2022.

Chen Gao, Liankai Jin, Xingyu Peng, Jiazhao Zhang, Yue Deng, Annan Li, He Wang, and Si Liu. Octonav: Towards generalist embodied navigation. arXiv preprint arXiv:2506.09839, 2025.

Jing Gu, Eliana Stefani, Qi Wu, Jesse Thomason, and Xin Eric Wang. Vision-and-language navigation: A survey of tasks, methods, and future directions. arXiv preprint arXiv:2203.12667, 2022.

Yanjia Huang, Mingyang Wu, Renjie Li, and Zhengzhong Tu. Vista: Generative visual imagination for visionand-language navigation. arXiv preprint arXiv:2505.07868, 2025.

Jacob Krantz, Erik Wijmans, Arjun Majumdar, Dhruv Batra, and Stefan Lee. Beyond the nav-graph: Visionand-language navigation in continuous environments. In European Conference on Computer Vision, pp. 104–120, 2020.

Alexander Ku, Peter Anderson, Roma Patel, Eugene Ie, and Jason Baldridge. Room-across-room: Multilingual vision-and-language navigation with dense spatiotemporal grounding. arXiv preprint arXiv:2010.07954, 2020.

Bingqian Lin, Yunshuang Nie, Khun Loun Zai, Ziming Wei, Mingfei Han, Rongtao Xu, Minzhe Niu, Jianhua Han, Liang Lin, Cewu Lu, et al. Evolvenav: Self-improving embodied reasoning for llm-based visionlanguage navigation. arXiv e-prints, pp. arXiv–2506, 2025a.

Bingqian Lin, Yunshuang Nie, Ziming Wei, Jiaqi Chen, Shikui Ma, Jianhua Han, Hang Xu, Xiaojun Chang, and Xiaodan Liang. Navcot: Boosting llm-based vision-and-language navigation via learning disentangled reasoning. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2025b.

Khanh Nguyen, Debadeepta Dey, Chris Brockett, and Bill Dolan. Vision-based navigation with language-based assistance via imitation learning with indirect intervention. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 12527–12537, 2019.

Xinshuai Song, Weixing Chen, Yang Liu, Weikai Chen, Guanbin Li, and Liang Lin. Towards long-horizon vision-language navigation: Platform, benchmark and method. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 12078–12088, 2025.

Keyu Tian, Yi Jiang, Zehuan Yuan, Bingyue Peng, and Liwei Wang. Visual autoregressive modeling: Scalable image generation via next-scale prediction. Advances in neural information processing systems, 37:84839– 84865, 2024.

Hanqing Wang, Wei Liang, Luc V Gool, and Wenguan Wang. Towards versatile embodied navigation. Advances in neural information processing systems, 35:36858–36874, 2022.

Shuo Wang, Yongcai Wang, Wanting Li, Xudong Cai, Yucheng Wang, Maiyue Chen, Kaihui Wang, Zhizhong Su, Deying Li, and Zhaoxin Fan. Aux-think: Exploring reasoning strategies for data-efficient vision-language navigation. Advances in Neural Information Processing Systems, 2025.

Xin Wang, Qiuyuan Huang, Asli Celikyilmaz, Jianfeng Gao, Dinghan Shen, Yuan-Fang Wang, William Yang Wang, and Lei Zhang. Vision-language navigation policy learning and adaptation. IEEE transactions on pattern analysis and machine intelligence, 43(12):4205–4216, 2020.

Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc V Le, Ed H Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. Self-consistency improves chain of thought reasoning in language models. In International Conference on Learning Representations, 2023.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824–24837, 2022.

Qiaoyun Wu, Xiaoxi Gong, Kai Xu, Dinesh Manocha, Jingxuan Dong, and Jun Wang. Towards target-driven visual navigation in indoor scenes via generative imitation learning. IEEE Robotics and Automation Letters, 6(1):175–182, 2020.

Wansen Wu, Tao Chang, Xinmeng Li, Quanjun Yin, and Yue Hu. Vision-language navigation: a survey and taxonomy. Neural Computing and Applications, 36(7):3291–3316, 2024.

Zifan Xu, Bo Liu, Xuesu Xiao, Anirudh Nair, and Peter Stone. Benchmarking reinforcement learning techniques for autonomous navigation. In ICRA, 2023.

Xinda Xue, Junjun Hu, Minghua Luo, Xie Shichao, Jintao Chen, Zixun Xie, Quan Kuichen, Guo Wei, Mu Xu, and Zedong Chu. Omninav: A unified framework for prospective exploration and visual-language navigation. arXiv preprint arXiv:2509.25687, 2025.

Kuo-Hao Zeng, Zichen Zhang, Kiana Ehsani, Rose Hendrix, Jordi Salvador, Alvaro Herrasti, Ross Girshick, Aniruddha Kembhavi, and Luca Weihs. Poliformer: Scaling on-policy rl with transformers results in masterful navigators. In Conference on Robot Learning, pp. 408–432. PMLR, 2025.

Jiazhao Zhang, Kunyu Wang, Rongtao Xu, Gengze Zhou, Yicong Hong, Xiaomeng Fang, Qi Wu, Zhizheng Zhang, and He Wang. Navid: Video-based vlm plans the next step for vision-and-language navigation. Robotics: Science and Systems, 2024a.

JiazhaoZhang, KunyuWang, ShaoanWang, MinghanLi, HaoranLiu, SonglinWei, ZhongyuanWang, Zhizheng Zhang, and He Wang. Uni-navid: A video-based vision-language-action model for unifying embodied navigation tasks. Robotics: Science and Systems, 2025a.

Wenyao Zhang, Hongsi Liu, Zekun Qi, Yunnan Wang, XinQiang Yu, Jiazhao Zhang, Runpei Dong, Jiawei He, He Wang, Zhizheng Zhang, et al. Dreamvla: A vision-language-action model dreamed with comprehensive world knowledge. In Advances in Neural Information Processing Systems, 2025b.

Zhuosheng Zhang, Aston Zhang, Mu Li, George Karypis, Alex Smola, et al. Multimodal chain-of-thought reasoning in language models. Transactions on Machine Learning Research, 2024b.

Zhuosheng Zhang, Aston Zhang, Mu Li, Hai Zhao, George Karypis, and Alex Smola. Multimodal chain-ofthought reasoning in language models. Transactions on Machine Learning Research, 2024, 2024c.

QingqingZhao, YaoLu, MooJinKim, ZipengFu, ZhuoyangZhang, YechengWu, ZhaoshuoLi, QianliMa, Song Han, Chelsea Finn, et al. Cot-vla: Visual chain-of-thought reasoning for vision-language-action models. In Proceedings of the IEEE/CVF Computer Vision and Pattern Recognition Conference, pp. 1702–1713, 2025.

Boyang Zheng, Nanye Ma, Shengbang Tong, and Saining Xie. Diffusion transformers with representation autoencoders. arXiv preprint arXiv:2510.11690, 2025.

Denny Zhou, Nathanael Sch¨arli, Le Hou, Jason Wei, Nathan Scales, Xuezhi Wang, Dale Schuurmans, Claire Cui, Olivier Bousquet, Quoc V Le, et al. Least-to-most prompting enables complex reasoning in large language models. In International Conference on Learning Representations, 2023.

Gengze Zhou, Yicong Hong, Zun Wang, Xin Eric Wang, and Qi Wu. Navgpt-2: Unleashing navigational reasoning capability for large vision-language models. In European Conference on Computer Vision, pp. 260–278. Springer, 2024.

## Appendix

- A Data Preparation

- A.1 Preprocessing

An expert trajectory in LH-VLN comprises a temporal sequence of visual–action pairs {ot,at}1:T together with the natural-language task instruction I. Since VLN is an online sequential decision-

making problem with continual interaction with the environment, the agent πθ must act in real time based on its current and historical visual observations {o≤t}. To construct training samples, we partition each navigation trajectory Ji into non-overlapping slices. Each slice contains the task instruction I, visual observations {o≤t}, and the future k actions At:

{I,{o≤t},At}t∈Si ∼ Slice(Ji), i = 1,...,N, (11)

where At = {at,at+1,...,at+k−1}, Si = {1,1+k,...,Ti}, Ti is the number of actions in Ji, and N is the number of navigation trajectories in LH-VLN. In practice, we set k = 5.

- A.2 Navigation Prompt

We use the following prompts to enable instruction-driven navigation behaviors across different reasoning modes. For single-stage task, we set the prompt as:

You are an autonomous navigation robot. You will get a task with historical and current pictures you see. Based on this information, you need to decide your next 5 actions, which could involve <|left|>, <|right|>,<|forward|>. If you finish your mission, output <|stop|>. Your current observations are left side: <image>, front side: <image>, right side: <image>. Your historical pictures are: <image> ... <image>. Your mission is: [instruction].

In multi-stage navigation tasks, the agent must stop upon completing each subtask and maintain awareness of how many subtasks have been finished. To this end, we further extend the prompt with the following description:

PS: The mission is complex. You may infer several subtasks within the mission, and output <|stop|> when a sub-task is achieved. So far, you have output <|stop|> 0 times. Historical information reflects progress up to the current subgoal.

- A.3 T-CoT Data Annotation

We employ Qwen-VL-Max to generate T-CoT annotations for each navigation slice (see Eq. (11)). All 18,554 navigation slices from the LH-VLN training set are annotated. The annotation prompt is as follows:

You are a professional AI data annotator. Your task is to label the intermediate CoT reasoning for VLN trajectory slices. Input includes: user instruction, ≤20 historical images (front view), 3 current images (left, front, and right view), and the GT action sequence.

Annotation steps:

- 1. Semantic Planning: Break the mission into precise sub-tasks with clear spatial goals (e.g., “reach the cabinet table in the living room,” “exit the living room,” “enter the office,” “approach the office table”). Sub-tasks should reflect stepwise navigation milestones rather than abstract summaries.
- 2. Visual Description: Based on the semantic plan, describe what the historical and current images reveal about completed and upcoming sub-tasks. Be concise, factual, and avoid vague wording.
- 3. Action Decision-Making: Predict the next action sequence (5 steps) aligned with the semantic plan and supported by the visual description. Provide a brief justification for the predicted steps. Use natural, instruction-like phrasing instead of raw action codes.
- 4. Visual Imagination: Describe the expected scene after executing the predicted actions, focusing on landmarks or key objects that should become visible or reachable.

# Output format: <think> Semantic Plan: ··· Visual Description: ··· Action Planning: ··· Visual Imagination: ··· </think> Note: Replace “···” with actual CoT content. Do not output quotes or ellipses. # User Instruction: [Instruction] # GT Action: <|forward|> ··· <|right|>

- A.4 Data Augmentation

To improve the robustness of the instruction-following model under diverse visual histories, we augment each training example by perturbing only the historical image sequence while keeping the final three observation images unchanged. Given an original sample with N historical frames {h1,...,hN} and the last three observation frames {oℓ,of,or}, we generate up to two additional augmented variants per sample. The augmentation operations are stochastic and applied with independent Bernoulli trials.

Uniform Subsampling. For trajectories with at least ten historical images, we apply uniform subsampling with probability 0.5. Specifically, we replace the original history {hi}Ni=1 with a stride2 subsequence {h1,h3,h5,...} while preserving the three observation images. This operation reduces temporal redundancy and encourages the model to rely on coarser but more informative state transitions.

Stochastic History Trimming. For trajectories with at least seven historical frames, we further apply stochastic trimming. Two perturbations may occur: (i) with probability 0.5, we remove the first two historical frames, yielding {h3,h4,...,hN}; (ii) with probability 0.5, and only when the remaining length is at least seven, we randomly select an index k and remove two consecutive frames {hk,hk+1}. At least one of the above operations must be triggered for the augmented sample to be retained. This procedure introduces temporal uncertainty, forcing the model to rely on stable, task-relevant cues rather than positional biases.

- B Detail Formulations.

Here, we provide the formal formulation of the proposed FantasyVLN, which unifies Non-CoT, T-CoT, V-CoT, and MM-CoT reasoning modes within a single reasoning framework. We introduce binary gating signals gT and gV to enable flexibly switching among the four reasoning modes.

Non-CoT Reasoning. Given the task instruction I and visual observations {o≤t}, the non-CoT reasoning mode aims to directly predict actions At based on the instruction I and observations {o≤t}:

###### At ∼ πθ I,{o≤t},gT = 0,gV = 0 , (12)

where πθ is the navigation agent. We employ a pretrained VLM as the navigation agent πθ and transfer its multimodal prior to the navigation task through supervised fine-tuning (SFT):

LCE At,At , (13) where LCE denotes the causal cross-entropy loss and At represents the ground truth actions.

arg min

θ

T-CoT reasoning. Textual CoT reasoning generates intermediate reasoning steps Tt before action prediction:

###### [ Tt, At] ∼ πθ I,{o≤t},gT = 1,gV = 0 . (14)

We define the intermediate reasoning steps Tt as a structured chain of thought that guides the navigation process. The agent πθ first decomposes the instruction I into a sequence of subgoals, and then infers the current goal from the visual observations {o≤t}. Finally, πθ identifies the decision evidence from the current visual observation ot. The training objective is formulated as:

LCE Tt,Tt + LCE At,At , (15) where Tt is the ground truth of textual reasoning steps.

arg min

θ

CompV-CoT reasoning. Visual CoT reasoning enhances spatial understanding by generating future visual observations Vt, which serve as the conditions for action prediction. We propose CompVCoT that conducts V-CoT in the latent space of a VAR model. Instead of producing pixel-level images, the agent πθ predicts low-scale latent representations of VAR Ht:

###### [ Ht, At] ∼ πθ I,{o≤t},gT = 0,gV = 1 . (16)

The predicted representations are then passed through the VAR model to reconstruct pixel observations:

Vt ∼ g Ht , (17) where g denotes the generation pipeline based on next-scale prediction for VAR. During training, the VAR is frozen. The training process is defined as:

LCE Vt,Vt + LCE At,At , (18) where Vt is the ground truth future visual observations. MM-CoT reasoning We employ paired textual-visual CoT reasoning steps as the villain Multimodal CoT reasoning steps Mt:

arg min

θ

###### Mt = [ Tt, Ht]. (19)

MM-CoT reasoning first generates multimodal reasoning steps Mt and then predicts future actions At:

[ Mt, At] ∼ πθ I,{o≤t},gT = 1,gV = 1 . (20) The training objective is formulated as:

LCE Mt,Mt + LCE At,At . (21)

arg min

θ

UM-CoT reasoning. FantasyVLN is a unified multimodal CoT reasoning framework that integrates the non-CoT, T-CoT, V-CoT, and MM-CoT reasoning modes. The formulation of UM-CoT is provided in Section 3.4.

- C Implementation Details

Training Details. We adopt Qwen2.5-VL as the base model and apply LoRA-based parameterefficient tuning to both the language layers and the vision–language projection modules. Training is conducted on 64 H20 GPUs, each equipped with 141GB of memory. We use the AdamW optimizer with a learning rate of 1 × 10−4, a weight decay of 0.1, and a cosine schedule with a

- 5% warmup ratio. The per-device batch size is set to 4, supported by 32 dataloader workers. We employ bfloat16 precision, enable gradient checkpointing, and adopt DeepSpeed ZeRO-2 for memory-efficient optimization.

Evaluation. We perform online evaluation for all methods. Given an initial position, the agent interacts with the simulator to execute multi-stage navigation tasks. At each step, the agent receives visual observations and predicts subsequent actions; the environment then applies the predicted action and updates the agent’s state. When the agent outputs <|stop|>, the environment verifies whether the current subtask is completed (i.e., the distance to the target location is within 1m) and then proceeds to the next subtask. If the agent exhausts its action budget before completing the subtask, the subtask is marked as failed. This procedure continues until all subtasks are completed or terminated.

Special Tokens. Leveraging the vocabulary extensibility of autoregressive models, we implement the required functionalities through systematic vocabulary expansion. Specifically, we introduce (i) action tokens <|forward|>, <|left|>, <|right|>, <|stop|> for navigation action prediction; (ii) VAR latent tokens <|1|>–<|4096|> for CompV-CoT and MM-CoT latent-space visual reasoning; (iii)systemtokenssuchas<|NAV|>, <think>, </think>toregulatenarrativeformatting; and (iv) gating tokens <textual think>, <no textual think>, <visual think>, <no visual think> that serve as the binary signals gT and gV for unified multimodal CoT control.

