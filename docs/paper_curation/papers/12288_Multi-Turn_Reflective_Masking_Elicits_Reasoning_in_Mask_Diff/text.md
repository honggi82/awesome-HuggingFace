# arXiv:2606.16700v1[cs.CL]15Jun2026

## Multi-Turn Reflective Masking Elicits Reasoning in Mask Diffusion Models

Yanming Zhang1∗ Yihan Bian1∗ Jingyuan Qi2 Yuguang Yao3 Lifu Huang4 Tianyi Zhou5 1University of Maryland 2Virginia Tech 3Intuit 4UC Davis 5MBZUAI

Project Page

### Abstract

While reasoning on autoregressive (AR) models is often performed by chain-ofthought reasoning and reflection, their refinement of previous outputs still relies on fully sequential generation, even when only local edits are needed. In contrast, the masking mechanism in Mask Diffusion Models (MDMs) naturally supports explicit local edits on previous outputs, allowing selective refinement without discarding previous answers and generating another from scratch. While this property more closely aligns with how humans correct mistakes by iterative local refinement, existing MDMs do not support multi-turn masking and denoising. We propose Reflective Masking (RM), which elicits such an intrinsic reasoning capability in MDMs via lightweight post-training. RM provides a native test-time scaling, where an MDM iteratively revisits and revises its prior outputs based on evolving context. To exploit insights from previous turns like AR reasoning, we further introduce History Reference, a parameter-free mechanism that leverages intermediate denoising states during revision. Our approach requires no architectural changes and is easily applicable to existing MDMs. Across diverse tasks and modalities, including text generation, Sudoku, and image editing, Reflective Masking consistently outperforms standard masking-based baselines and demonstrates strong generality, positioning RM as a fundamental primitive for reasoning on MDMs.

### 1 Introduction

While Large Language Models (LLMs) have been widely accepted for performing various reasoning tasks [21, 41, 25, 43], recent studies have revealed that they still struggle in multi-turn and longhorizon settings [22]: models tend to propagate previous errors and incorrect assumptions rather than revise them. A key limitation of their autoregressive (AR) paradigm is that local errors require regenerating the entire sequence, incurring unnecessary computation. Moreover, incorrect intermediate content persists in the context, occupying valuable capacity and contaminating subsequent reasoning.

In contrast, Mask Diffusion Models (MDMs) [2, 33, 31, 47] provide a fundamentally different generation mechanism that naturally supports localized revision. Through iterative masked updates, MDMs can keep the current context fixed while resampling uncertain tokens, avoiding the need to regenerate the entire sequence. This native remasking mechanism reveals a potential advantage of MDMs for reasoning: if local errors can be revised in place, the model could maintain a cleaner intermediate context, reduce error contamination, and save computation. In this sense, MDMs offer a possible path toward localized self-correction that is difficult to realize in AR generation.

However, realizing this potential requires moving beyond the passive remasking behavior of existing MDM decoding. Standard MDM generation still follows an absorbing Markovian denoising process:

∗Equal contribution.

Preprint.

once a token is confidently denoised, it is fixed. Existing remasking strategies focus on re-masking low-confidence tokens, but the model is unable to actively revisit and correct previously committed predictions. We argue that for MDMs to support reasoning-level self-correction and provide a distinct form of test-time scaling from AR models, the model must learn to identify unreliable predictions and actively revise them during generation. We propose Reflective Masking, where masking is formulated as an internal decision process driven by the model’s uncertainty, enabling it to selectively refine already denoised tokens during generation.

In this work, we activate Reflective Masking as a first-class capability of MDMs by making it a central focus of post-training. We introduce a lightweight training paradigm that enables selfinitiated, context-aware revision without architectural modifications, together with an effective data generation strategy that produces stable training signals aligned with the model’s native output distribution. Our results show that this latent capability can be reliably unlocked through appropriate training. More broadly, reflective masking establishes a form of test-time scaling unique to MDMs, where additional computation is allocated to selective revision rather than forward expansion. We argue that reasoning in MDMs should be viewed not as forward generation, but as iterative state refinement through explicit self-correction and revision.

With multi-turn reflective masking, a key capability still missing relative to autoregressive reasoning is the ability to extract insights from the growing context of historical generations. To bridge this gap, we introduce History Reference, a parameter-free mechanism that enables MDMs to maintain a stateful view of their denoising trajectory by preserving intermediate decoding states. Unlike AR models, which encode history implicitly in the input context, History Reference allows MDMs to explicitly leverage past predictions. This mechanism introduces a temporal dimension orthogonal to the current text context, guiding reflective updates and improving consistency while reducing repeated errors during iterative refinement. Importantly, History Reference requires no additional learnable parameters or external memory, making it efficient and readily applicable to existing MDMs.

We evaluate our approach across a spectrum of generation tasks with varying levels of guidance, demonstrating consistent improvements and strong generality. We begin with image editing, where rich instructions specify both where and how to modify the input. We then consider Sudoku, a structured reasoning task that requires the model to identify and correct erroneous entries. Finally, we study text generation tasks, where supervision is minimal and no direct hints about the final answer are provided. Across all three settings, Reflective Masking consistently improves performance. For tasks that require autonomous exploration, History Reference proves particularly effective in guiding the model’s reasoning and iterative revision. Together, these results suggest that reflective masking provides a general mechanism for enabling reasoning through explicit revision in mask-based generation.

Our contributions can be summarized as follows:

- • We identify Reflective Masking as a latent capability of mask diffusion models, and propose a new perspective that frames generation as an iterative process of self-initiated revision, introducing a new dimension of reasoning based on explicit modification of prior outputs.
- • We propose a lightweight training paradigm, associated with an effective and scalable data pipeline, to activate Reflective Masking as an intrinsic skill, enabling context-aware and adaptive revision without architectural changes. We demonstrate its effectiveness across diverse downstream tasks and modalities, including text generation, Sudoku, and image editing, highlighting strong generality.
- • We introduce History Reference, a parameter-free mechanism that enables MDMs to maintain a stateful view of their denoising trajectory by incorporating intermediate decoding history. It improves revision consistency and avoids repeating past errors during iterative refinement.

### 2 Related Work

Mask diffusion models. MDMs [31, 48, 27] generate sequences by iteratively denoising masked inputs [11, 6, 24, 13]. Lumina-DiMOO [47] extends this to a multimodal setting. While these formulations naturally allow tokens to be revisited through masking [14, 46], existing approaches primarily focus on one-shot denoising and do not exploit this capability for iterative revision.

Editing and revision in autoregressive models. Autoregressive models have also been extended to support revision and editing of generated content. Prior work explores editing capabilities by inserting special tokens or performing span-level regeneration, such as insertion-based generation [36, 15],

edit-based modeling [16], and controllable text editing frameworks [30, 28, 29, 10]. However, such mechanisms are not native to autoregressive generation, which fundamentally operates in a forwardonly manner. As a result, revising prior outputs typically requires regenerating entire sequences, introducing additional decoding passes. This makes editing indirect and often inefficient, as the model cannot modify earlier decisions in place. In contrast, mask diffusion models naturally support in-place modification through masking, enabling direct and localized revision of generated content. This structural difference provides a more suitable foundation for iterative self-correction, where previous predictions can be selectively revisited and refined without restarting the generation process.

RemeDi and re-masking approaches. RemeDi [19] introduces self-reflective re-masking with a dual-stream architecture and achieves strong performance on from-scratch text generation, highlighting the importance of revising intermediate predictions in mask diffusion language models. However, RemeDi treats re-masking as an additional capability that must be explicitly learned through architectural modifications and auxiliary training objectives. This design increases both training and inference complexity, and limits its adaptability to existing MDM frameworks.

Other works explore variants of re-masking, including mixed noise schedules [38], predictor–corrector strategies [39], and per-step resampling schemes [35]. These approaches improve sampling efficiency or stability, but continue to treat masking as an externally driven procedure rather than an intrinsic model behavior. In contrast, we view masking as a native capability of MDMs that can be activated rather than newly introduced, reframing revision as an internal decision process rather than an externally imposed operation.

### 3 Reflective Masking

Reflective Masking (RM) treats every position at every denoising step as a per-position decision: keep the current token, re-mask it to MASK for re-prediction, or, if currently masked, reveal a new token. We present RM in the following parts: the inference decision rule that maps model probabilities to per-position keep / re-mask / reveal actions (§3.1); History Reference (HR), a parameter-free perposition history aggregation mechanism that summarizes the denoising trajectory into a history-aware embedding for the model (§3.2); the training objective that enables models’ Reflective Masking capability and the recipe used to construct training inputs (§3.3).

Notations. Let x∗ ∈ VN denote the target sequence and E ⊆ {1,...,N} the set of editable positions (full sequence excluding the instruction prompt). For each i ∈ E, the position can take one of three

values throughout the denoising trajectory: x˜(it) ∈ {wi,MASK,x∗i } where wi is a task-specific wrong token. Non-edit positions i ∈/ E stay at x∗i at every step. We write T for the total number of denoising steps and V¯ := V ∪ {MASK} for the extended vocabulary.

#### 3.1 Basic inference rules on MDMs with Reflective Masking enabled

Unlike the standard MDM inference rule, which follows an absorbing Markov process, we introduce Reflective Masking that allows the model to revisit and revise past decisions, enabling test-time scaling. To enable this, we define a new inference rule. At timestep t ∈ {0,1,...,T − 1} the model takes the current state x˜(t) as input and outputs a per-position categorical pθ(· | x˜(t))i over V¯. The next-step state is then determined per position by a deterministic rule that splits on whether the position is currently non-mask or masked. MASK is denoted as M:

 

M if x˜(it) ̸= M and pθ(M | x˜(t))i > pθ(˜x(it) | x˜(t))i (RM), x˜(it) if x˜(it) ̸= M and pθ(M | x˜(t))i ≤ pθ(˜x(it) | x˜(t))i (Keep), arg max

x˜(it+1) =

(1)



pθ(v | x˜(t))i if x˜(it) = M (Reveal).

v∈V

The rule is intuitive. At a non-mask position, re-mask whenever the model assigns higher probability to MASK than to the current token, because the model is signaling that the non-mask token is wrong; otherwise keep it. At a MASK position, reveal the most likely vocabulary token. In general, at each iteration, model can produce one of three actions per position: keep, re-mask, or reveal, all driven entirely by the model’s per-position output distribution.

This basic rule as presented conditions only on the current state x˜(t). As a result, the model may produce the same state across different time steps, leading to a loop where identical states are repeatedly passed to subsequent steps. We address this in §3.2 by changing what the model sees as input, while leaving Eq. (1) unchanged.

#### 3.2 Enhancing Reflective Masking with History Reference

To let the model condition on more than the current state, we maintain a per-position accumulated embedding

that compresses the prefix x˜(0:i t) into a single vector and feed it to the model.

The accumulated embedding adds no learnable parameters and admits an O(1) per-step update (Appendix B).

Setup. Let e(ik) := wte(˜x(ik)) denote the model token embedding of

position i at step k. Let R∆ denote the History Embedding Rotation (HER) indexed by the distance ∆ = k − t between a historical step k and the current step t, satisfying the standard rotation composition rules

𝑥𝑥(t-i) 𝑥𝑥(t-1) 𝑥𝑥(t) 𝑥𝑥(t+1)

2

2 M M a M 0

2 M 0 a M 0

2 a M a = 0

b M a M M

M D M

HR

HER

Correct Mask Wrong Embedding

Figure 1: Inference procedure with History Reference. All states are embedded, and the historical states are further processed by HER. The embedding of current step x˜(t) is added to the history reference (HR) and fed into the model as the input, which predicts the next step x˜(t+1).

R0 = I, RaRb = Ra+b, R−∆ = R∆⊤.

R∆ is a parameter-free block-diagonal rotation that applies a distance to each two-dimensional block of an embedding. We instantiate it with the same two-dimensional sinusoidal blocks used by standard rotary encodings [37].

History embedding. At iteration t we express every historical state in the current step’s reference frame and accumulate them into a single vector:

t

γ t−k Rk−t e(ik), (2)

a(it) =

k=0

where γ ∈ (0,1] is a history-decay factor. Equivalently,

a(it) = e(it) + γ R−1 e(it−1) + γ2 R−2 e(it−2) + ··· + γt R−t e(0)i . The current state contributes unrotated, while each past state is added with a lag-dependent rotation R∆ and decay γ|∆|. Two sequences that share the same current state are augmented by their histories, which provide additional conditioning signals. This historical context allows the model to reference past states when deciding whether to re-mask or keep a token, helping it avoid recurring errors and leverage useful cues from earlier versions.

#### 3.3 Training paradigm for enabling Reflective Masking in MDMs

To enable the model to follow the inference rule of §3.1–§3.2, we design a training paradigm that aligns its per-position outputs with the correct next-step actions. Concretely, we train the model with per-position oracle labels, encouraging its output distribution to assign the highest probability to the desired next action at each position.

Oracle revision rule. For each i ∈ E and any state zi ∈ {wi,MASK,x∗i }, the optimal next-step action is the deterministic function

 

MASK zi ∈ V \ {x∗i } (re-mask wrong / initial token)

τ(zi,x∗i ) =

(3)

x∗i zi = MASK (reveal target) x∗i zi = x∗i (preserve target).



Sample noisy positions

History construction rules

2 a - 2 a = 0

Original x*

Correct

c0 m0 w0

cT

ct mt wt

2 M - 2 M = 0

Sample mask

Mask Wrong

mT wt1 wt2 wT

mt1

Mask Diffusion Model

MDM Forward

Top-k at remaining noisy positions

2 M - 0 M ) 0

3

) + -

History Reference

Sample wrong

- 0

- 1

2 M - 0 a M 0 Current Step

2 M - 0 M ) 0

###### Train Loss

Reveal Loss RM Loss Keep Loss

Correct M Mask Wrong Noisy

2 a - M a = 0 Train target

- Figure 2: Synthetic history data construction for Mask Diffusion Model Training. A noisy sequence is created from a clean sequence using mask and wrong-token corruption. Position-wise transition rules are then used to sample synthetic histories and define training targets.

A non-mask token that differs from the target should be re-masked, a MASK should be revealed to the target, and a correct non-mask token should be preserved. Non-edit positions are deterministically labelled x∗i at every step.

Training data curation. As illustrated in Fig. 2, a training instance is constructed by simulating a synthetic trajectory, starting from an original clean sequence x∗. We first sample a subset of positions to corrupt and draw a timestep t ∼ Uniform{0,...,T − 1}. At this timestep, the selected positions are split into two groups: one is replaced with MASK tokens, while the other is substituted with wrong tokens. Wrong tokens are sampled from a corruption distribution ν(· | x∗i ) over V \ {x∗i }. In practice, ν is chosen to better match inference-time errors, e.g., using top-k predictions from a frozen MDM backbone (excluding the ground-truth token) or task-specific token distributions. Appendix A.4 bounds the resulting shift of the training objective relative to a uniform proposal. This yields the current state z = x˜(t).

To approximate inference-time dynamics, we construct histories following position-wise transition rules (right side of Fig. 2). Each position evolves independently according to its current state: correct tokens remain unchanged; masked tokens transition to correct tokens at a sampled timestep t1; and wrong tokens first transition to mask tokens at sampled timestep t1 and then to correct tokens at later timestep t2. These transitions are governed by per-position rules together with sampled transition timesteps within the T-step horizon, which jointly define the trajectory sampler.

Sampling these transition timesteps produces a set of reference states that mimic iterative refinement during inference. Based on the resulting state, training targets are defined per position: masked tokens are trained to predict the correct token, while wrong tokens are trained to predict the mask token. The overall objective combines reveal, mask, and keep losses. This construction defines a trajectory sampler that specifies the training input distribution, while remaining independent of model parameters θ, providing a stable target for optimization. The full procedure is given in Appendix C.

Training objective. The model receives a(t) and outputs per-position categorical distributions. We supervise each position using an oracle action τ(zi,x∗i ) that depends on its current state: masked tokens are mapped to their correct token (reveal), wrong tokens are mapped to MASK (mask), and correct tokens are kept unchanged (keep).

We minimize the per-position cross-entropy against this oracle,

−log pθ τ(zi,x∗i ) a(t)

, (4)

Ltrain(θ) = E

i

i∈E

where the expectation is over the data distribution, the random trajectory and step sampling.

Equivalently, the objective decomposes into three components: a reveal loss, which predicts the correct token for masked positions; a mask loss, which predicts MASK for wrong positions; and a keep loss, which preserves correct tokens. This decomposition matches the training targets induced by the synthetic trajectory construction in Fig. 2.

Two properties of this objective justify its use. First, given a sufficiently expressive model, minimizing Eq. (4) drives the model’s per-input output toward the conditional distribution of the oracle action τ, so the argmax of the trained model recovers the inference rule of Eq. (1). Second, conditioning on a(t) instead of the bare current state x˜(t) cannot increase the optimal training risk, since a richer input representation can only improve (or leave unchanged) the best achievable expected loss. Please refer to Appendix A for the complete theoretical justification and detailed proofs.

### 4 Experiments

We evaluate our approach across three representative task categories that differ in the level of external guidance and the degree of required autonomous reasoning. These tasks are designed to systematically probe how Reflective Masking and History Reference contribute under varying conditions of supervision and exploration. At one end of the spectrum, we consider image editing, where rich instructions directly specify both the target regions and desired modifications. We then study Sudoku, a structured reasoning task where the model must identify and correct erroneous entries with limited guidance. Finally, we evaluate on text reasoning tasks such as mathematical reasoning and code synthesis, where no explicit hints about the final output are provided. This progression enables us to examine the roles of RM and HR, and to understand how they interact across tasks with increasing reasoning complexity and decreasing supervision.

The training on these three tasks can be completed within about 5 hours on 2 NVIDIA H100 80GB GPUs, avoiding the days or weeks long training often associated with prior architecture-changed approaches.

#### 4.1 Instruction-based in-place image editing

We begin with image editing [5, 50, 34, 18], a task characterized by strong external guidance from natural language instructions. Given an input image and an editing instruction, the model is required to localize the regions to be modified and generate corresponding updates that align with the semantic intent, while preserving the rest of the image. In this setting, the instruction provides explicit cues for both where and how to edit, leaving limited need for autonomous exploration. As a result, this task primarily evaluates whether Reflective Masking can enable precise, localized revisions based solely on instruction signals, without relying on iterative self-discovery or history-based reasoning.

Experiment settings. We use Lumina-DiMOO (Lumina) [47] as the base model. For training and evaluation, we sample 85,000 examples from ImgEdit [49] as the training set and hold out another 1,700 examples as the test set. To ensure a fair comparison, our method and the vanilla SFT baseline are trained under the same experimental setup.

Table 1: Quantitative results on the image editing task.

Edit Localization Background Preservation Overall Editing Quality Prec. ↑ Coverage ↑ MAE-RGB ↓ PSNR (dB) ↑ SSIM ↑ VQAScore ↑ User Study ↑

Method

Lumina 65.81 41.68 12.497 23.092 0.6256 71.95 41.8 Lumina+SFT 71.84 48.42 11.035 23.900 0.6570 81.61 53.3 RM (Ours) 99.73 73.02 3.613 34.759 0.9744 85.17 68.2

- Results. Fig. 3 and Tab. 1 present the qualitative and quantitative results on image editing. As shown in Fig. 3, our method accurately localizes the regions that require editing and masks them for subsequent generation. This provides the MDM with a cleaner current state, leading to higher-quality edits than the baselines. Moreover, since our method only regenerates the masked regions, the unmasked areas remain well aligned with the input image. This is further demonstrated by the heat maps in Fig. 3, where our changes are concentrated in the target edit regions, while the baselines

###### Instruction Original Predict Mask(Ours) Ours Lumina Lumina (SFT)

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

Turn time into black …

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

Add a person in wedding gown …

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

Add a cabin made from wood…

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

Turn cookies into chocolate …

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

Turn trees on the left into orange …

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

Add bell-shaped blooms …

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

Add creamy … filling a spoon …

###### Instruction Original Ours Lumina Lumina(SFT)

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

Turn the tree into snowy …

- Figure 3: Qualitative results on the image editing task. The masks predicted by RM are highlighted in red. Heat maps at the bottom visualize pixel-wise differences between the edited images and the originals. Please refer to Appendix D for the complete editing prompts and more qualitative results.

introduce more globally distributed editing noise, leading to noticeable consistency issues that distort unaffected regions and even corrupt fine-grained details.

For quantitative evaluation, we consider three aspects of image editing quality: localization accuracy, background preservation, and overall editing quality. For localization, we report Edit Precision, which measures the percentage of changed pixels that fall inside the ground-truth target edit region, while Edit Coverage measures the fraction of the target region that is modified. For background preservation, we use MAE-RGB [45], PSNR [20], and SSIM [42] on the non-target regions between the edited and the input image. These metrics evaluate whether regions unrelated to the instruction remain unchanged. For overall editing quality, we use VQAScore [26] to evaluate instruction following and content preservation, together with a 29-participant user study to measure human preference. As shown in Tab. 1, our method outperforms baselines across all metrics.

#### 4.2 Sudoku revision

We next consider Sudoku [32, 40], a structured reasoning task that requires the model to detect and correct errors in a partially incorrect grid. The input consists of Sudoku boards containing invalid entries, and the model must iteratively refine its predictions to produce a valid solution. Unlike image editing, this task requires the model to actively identify inconsistencies and explore possible corrections. At the same time, the search space is highly constrained due to the limited vocabulary and strict structural rules, making Sudoku a controlled environment for studying iterative reasoning and revision behaviors. This setting provides a challenging testbed for Reflective Masking and allows us to evaluate how History Reference facilitates effective failure avoidance in the absence of explicit instructions about where errors occur or how revisions should be performed, thereby highlighting the importance of the HR mechanism for autonomous error localization and refinement.

Table 2: Quantitative results on Sudoku revision.

Variant Exact Accuracy(%) ↑ Valid Rate(%) ↑ Replay Mistake(%) ↓ Conflict Cells (Number / board) ↓

RM w/o HR 82.4 86.6 0.57 0.578 RM + HR 91.4 (↑ 9.0) 91.8 (↑ 5.2) 0.07 (↓ 0.50) 0.300 (↓ 0.278) RM + HR + decay 89.4 (↑ 7.0) 89.6 (↑ 3.0) 0.07 (↓ 0.50) 0.362 (↓ 0.216) RM + HR + decay + HER 93.4 (↑ 11.0) 93.6 (↑ 7.0) 0.03 (↓ 0.54) 0.236 (↓ 0.342)

Experiment settings. We construct a lightweight MDM with four Transformer layers and 0.81M parameters. We construct test examples from solved 9×9 boards by corrupting a specified number of cells (randomly chosen from 4 to 20), replacing the original digits with incorrect values. The model is tasked with correcting the corrupted board through iterative re-masking and re-prediction.

Results. We evaluate four complementary metrics. Exact Accuracy requires every cell in the final board to exactly match the ground-truth solution, while Valid Rate only requires the final board to satisfy Sudoku constraints. We include Exact Accuracy because our setting allows the model to re-mask and revise any position in the initial board. A low Exact Accuracy score would indicate that the model tends to discard key given clues and regenerate a generic valid solution. Taken together, these two metrics provide a more comprehensive evaluation of the correction capability of the RM method. Replay Mistake measures the fraction of re-masked erroneous cells that are later decoded back to their previous incorrect digit during the revision trajectory, and Conflict Cells reports the average number of Sudoku-rule-violating cells in each final board.

As shown in Tab. 2, adding HR greatly reduces repeated mistakes and constraint conflicts compared with the variant without HR, suggesting that History Reference helps the model avoid revisiting the same erroneous predictions. However, introducing a decay factor alone still improves over the no-history baseline but performs worse than the HR-only variant, indicating that simply weakening historical signals is insufficient for effective correction. After further incorporating HER, which explicitly disentangles historical information, applying decay becomes beneficial. Our full method achieves the best performance across all metrics. These results suggest that properly structuring historical information, rather than merely attenuating it, is crucial for improving iterative error correction, leading to more accurate Sudoku solutions with fewer rule violations.

#### 4.3 Text reasoning task

Finally, we evaluate on text generation tasks, including mathematical problem solving and code generation, which require fully autonomous reasoning. Given only a problem description, the model must generate complete solutions without any direct hints about the final answer. These tasks present the most challenging setting, as the model needs to explore an unconstrained output space while maintaining logical consistency. In this regime, both Reflective Masking and History Reference become critical: RM enables iterative revision of intermediate outputs, while HR provides essential guidance by leveraging prior predictions to stabilize and improve the reasoning process.

Experiment settings. We use LLaDA [31] as the base model and implement it using the dLLM [51] library. We evaluate our method on reasoning benchmarks including MATH [25, 17, 23], MBPP [3], and ARC-Challenge [8]. MATH and MBPP require multi-step reasoning and unconstrained generation, making them suitable for evaluating the revision capability introduced by RM. For each benchmark, we construct task-specific training data from the corresponding training split to equip the model with the revision capability of RM. For MBPP, we randomly use 30% of the data for training and reserve the remaining 70% for testing. In contrast, ARC-Challenge is evaluated in a standard forward-only multiple-choice setting, serving primarily to verify that introducing RM does not compromise the model’s original inference capability on conventional generation tasks.

Table 3: Performance comparison across benchmarks. ∆ denotes the improvement over Vanilla SFT.

Benchmark Category LLaDA(%) Vanilla SFT(%) Ours(%) ∆

MATH500 Math 19.4 22.4 24.8 ↑ 2.4 MBPP Code 28.0 30.6 39.4 ↑ 8.8 ARC-Challenge MCQA 73.7 81.3 86.1 ↑ 4.8

The solution to -4 < 2(x - 1) < 8 is expressed in the form a < x < b. Find the value of a + b.

[Figure 43]

Step 45

Step 115

Step 135

- Step 138

... a + b = - 2 + 4 = 2 ... Answer : $\ boxed { 2 }$ …

... a + b = - M + M = 2 ... Answer : $\ boxed { 2 }$ …

... a + b = - 1 + 5 = M ... Answer : $\ boxed { 2 }$ …

... a + b = - 1 + 5 = 4 ... Answer : $\ boxed { M }$ …

- Step 139 ... a + b = - 1 + 5 = 4 ... Answer : $\ boxed { 4 }$ …

[Figure 44]

- Figure 4: RM actively re-masks and corrects tokens during inference based on the evolving global context. As the preceding context is refined, the correction propagates to the final answer. Please refer to Appendix E for more results and representative revision trajectories.

- Results. Fig. 4 illustrates a representative revision trajectory, while Tab. 3 and Tab. 4 report the quantitative results on text reasoning tasks. As shown in Fig. 4, even when the model initially produces an incorrect answer, the introduction of RM enables subsequent correction through iterative revision. Specifically, the model first focuses on regions with richer local context, the chain-ofthought (CoT) portion, where it identifies erroneous tokens, selectively re-masks them, and predicts corrected replacements while avoiding unnecessary changes to already correct tokens. As the CoT becomes progressively more accurate, the improved context is propagated to the rest of the sequence. This allows the model to subsequently revise the final answer, ultimately arriving at the correct solution. Moreover, compared with prior re-mask methods that mainly involve fewer logic-critical token changes, our RM can revise mathematical tokens that directly determine the reasoning process and final answer. This behavior highlights that RM not only enables error correction but also performs targeted and context-aware revisions, improving both intermediate reasoning steps and final outputs.

Table 4: Performance comparison on Minerva MATH task across different subject categories.

Subject Algebra Counting & Prob. Geometry Intermediate Alg. Num Theory Prealgebra Precalc Aggregate

Vanilla SFT(%) 28.90 17.72 20.67 13.07 17.41 36.28 14.10 22.62 Ours RM (%) 29.49 18.35 20.67 14.40 21.67 38.00 16.67 24.10 ∆(%) ↑ 0.59 ↑ 0.63 0 ↑ 1.33 ↑ 4.26 ↑ 1.72 ↑ 2.57 ↑ 1.48

Tab. 3 shows that RM consistently improves over both LLaDA and Vanilla SFT across math, code, and ARC-Challenge benchmarks. Tab. 4 further breaks down the results on Minerva MATH [23] by subject category. RM improves over Vanilla SFT on nearly all subjects. Notably, the performance gain on MBPP is larger than that on MATH500. We attribute this to the nature of code generation tasks, where correctness depends on a larger number of tokens compared to MATH500. In contrast, MATH500 evaluation focuses primarily on the final answer. As a result, our RM method benefits more in code tasks, where iterative revision can correct a greater number of token-level errors. These results suggest that enabling the model to revise previous predictions provides consistent benefits for text generation, especially in tasks that require multi-step reasoning or structured outputs.

- 5 Limitations and future work

While we evaluate RM on image editing, Sudoku revision, and text generation, these tasks still remain substantially simpler than the most challenging long-horizon reasoning problems typically studied in AR models, due to the limited reasoning capability of the current base mask diffusion models. In addition, our experiments are constrained by computational resources, and we do not investigate whether the RM capability can effectively transfer under significantly larger-scale training regimes. For future work, applying RM to stronger block diffusion models [1, 7, 4] presents an interesting direction; however, a direct application would only revise tokens within the currently generated block, limiting its effectiveness. Developing mechanisms for revising the full context beyond the current block may therefore be important for enabling more global self-correction and long-range reasoning.

### 6 Conclusion

In this work, we present Reflective Masking, a lightweight post-training framework that elicits reflective revision as an intrinsic capability of Mask Diffusion Models. Instead of treating generation as a one-way denoising process, RM enables an MDM to iteratively revisit, re-mask, and refine its previous predictions based on the evolving context. We further introduce History Reference, a parameter-free mechanism that exposes intermediate denoising states to the model and helps stabilize multi-turn revision by reducing repeated errors. Experiments on image editing, Sudoku revision, and text generation suggest that explicit in-place revision provides a natural form of test-time scaling for MDMs, opening a promising direction for reasoning with diffusion-based generative models.

### References

- [1] Marianne Arriola, Aaron Gokaslan, Justin T Chiu, Zhihan Yang, Zhixuan Qi, Jiaqi Han, Subham Sekhar Sahoo, and Volodymyr Kuleshov. Block diffusion: Interpolating between autoregressive and diffusion language models. arXiv preprint arXiv:2503.09573, 2025.
- [2] Jacob Austin, Daniel D Johnson, Jonathan Ho, Daniel Tarlow, and Rianne Van Den Berg. Structured denoising diffusion models in discrete state-spaces. Advances in neural information processing systems, 34:17981–17993, 2021.
- [3] Jacob Austin, Augustus Odena, Maxwell Nye, Maarten Bosma, Henryk Michalewski, David Dohan, Ellen Jiang, Carrie Cai, Michael Terry, Quoc Le, et al. Program synthesis with large language models. arXiv preprint arXiv:2108.07732, 2021.
- [4] Tiwei Bie, Maosong Cao, Kun Chen, Lun Du, Mingliang Gong, Zhuochen Gong, Yanmei Gu, Jiaqi Hu, Zenan Huang, Zhenzhong Lan, et al. Llada2. 0: Scaling up diffusion language models to 100b. arXiv preprint arXiv:2512.15745, 2025.
- [5] Tim Brooks, Aleksander Holynski, and Alexei A Efros. Instructpix2pix: Learning to follow image editing instructions. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 18392–18402, 2023.
- [6] Huiwen Chang, Han Zhang, Lu Jiang, Ce Liu, and William T Freeman. Maskgit: Masked generative image transformer. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 11315–11325, 2022.
- [7] Shuang Cheng, Yihan Bian, Dawei Liu, Linfeng Zhang, Qian Yao, Zhongbo Tian, Wenhai Wang, Qipeng Guo, Kai Chen, Biqing Qi, et al. Sdar: A synergistic diffusion-autoregression paradigm for scalable sequence generation. arXiv preprint arXiv:2510.06303, 2025.
- [8] Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. Think you have solved question answering? try arc, the ai2 reasoning challenge. arXiv preprint arXiv:1803.05457, 2018.
- [9] Luc Devroye, László Györfi, and Gábor Lugosi. A probabilistic theory of pattern recognition, volume 31. Springer Science & Business Media, 2013.
- [10] Felix Faltings, Michel Galley, Gerold Hintz, Chris Brockett, Chris Quirk, Jianfeng Gao, and William B Dolan. Text editing by command. In Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 5259–5274, 2021.
- [11] Marjan Ghazvininejad, Omer Levy, Yinhan Liu, and Luke Zettlemoyer. Mask-predict: Parallel decoding of conditional masked language models. In Proceedings of the 2019 conference on empirical methods in natural language processing and the 9th international joint conference on natural language processing (EMNLP-IJCNLP), pages 6112–6121, 2019.
- [12] Tilmann Gneiting and Adrian E Raftery. Strictly proper scoring rules, prediction, and estimation. Journal of the American statistical Association, 102(477):359–378, 2007.

- [13] Shansan Gong, Shivam Agarwal, Yizhe Zhang, Jiacheng Ye, Lin Zheng, Mukai Li, Chenxin An, Peilin Zhao, Wei Bi, Jiawei Han, et al. Scaling diffusion language models via adaptation from autoregressive models. arXiv preprint arXiv:2410.17891, 2024.
- [14] Shansan Gong, Ruixiang Zhang, Huangjie Zheng, Jiatao Gu, Navdeep Jaitly, Lingpeng Kong, and Yizhe Zhang. Diffucoder: Understanding and improving masked diffusion models for code generation. arXiv preprint arXiv:2506.20639, 2025.
- [15] Jiatao Gu, Changhan Wang, and Junbo Zhao. Levenshtein transformer. Advances in neural information processing systems, 32, 2019.
- [16] Kelvin Guu, Tatsunori B Hashimoto, Yonatan Oren, and Percy Liang. Generating sentences by editing prototypes. Transactions of the Association for Computational Linguistics, 6:437–450, 2018.
- [17] Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset. arXiv preprint arXiv:2103.03874, 2021.
- [18] Amir Hertz, Ron Mokady, Jay Tenenbaum, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Prompt-to-prompt image editing with cross attention control. arXiv preprint arXiv:2208.01626, 2022.
- [19] Zemin Huang, Yuhang Wang, Zhiyang Chen, and Guo-Jun Qi. Don’t settle too early: Selfreflective remasking for diffusion language models. arXiv preprint arXiv:2509.23653, 2025.
- [20] Quan Huynh-Thu and Mohammed Ghanbari. Scope of validity of psnr in image/video quality assessment. Electronics letters, 44(13):800–801, 2008.
- [21] Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. Large language models are zero-shot reasoners. Advances in neural information processing systems, 35:22199–22213, 2022.
- [22] Philippe Laban, Hiroaki Hayashi, Yingbo Zhou, and Jennifer Neville. Llms get lost in multi-turn conversation. arXiv preprint arXiv:2505.06120, 2025.
- [23] Aitor Lewkowycz, Anders Andreassen, David Dohan, Ethan Dyer, Henryk Michalewski, Vinay Ramasesh, Ambrose Slone, Cem Anil, Imanol Schlag, Theo Gutman-Solo, et al. Solving quantitative reasoning problems with language models. Advances in neural information processing systems, 35:3843–3857, 2022.
- [24] Xiang Li, John Thickstun, Ishaan Gulrajani, Percy S Liang, and Tatsunori B Hashimoto. Diffusion-lm improves controllable text generation. Advances in neural information processing systems, 35:4328–4343, 2022.
- [25] Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harrison Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step. In The twelfth international conference on learning representations, 2023.
- [26] Zhiqiu Lin, Deepak Pathak, Baiqi Li, Jiayao Li, Xide Xia, Graham Neubig, Pengchuan Zhang, and Deva Ramanan. Evaluating text-to-visual generation with image-to-text generation. In European Conference on Computer Vision, pages 366–384. Springer, 2024.
- [27] Aaron Lou, Chenlin Meng, and Stefano Ermon. Discrete diffusion modeling by estimating the ratios of the data distribution. arXiv preprint arXiv:2310.16834, 2023.
- [28] Jonathan Mallinson, Aliaksei Severyn, Eric Malmi, and Guillermo Garrido. Felix: Flexible text editing through tagging and insertion. In Findings of the Association for Computational Linguistics: EMNLP 2020, pages 1244–1255, 2020.
- [29] Jonathan Mallinson, Jakub Adamek, Eric Malmi, and Aliaksei Severyn. Edit5: Semiautoregressive text editing with t5 warm-start. In Findings of the Association for Computational Linguistics: EMNLP 2022, pages 2126–2138, 2022.

- [30] Eric Malmi, Sebastian Krause, Sascha Rothe, Daniil Mirylenka, and Aliaksei Severyn. Encode, tag, realize: High-precision text editing. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 5054–5065, 2019.
- [31] Shen Nie, Fengqi Zhu, Zebin You, Xiaolu Zhang, Jingyang Ou, Jun Hu, Jun Zhou, Yankai Lin, Ji-Rong Wen, and Chongxuan Li. Large language diffusion models. arXiv preprint arXiv:2502.09992, 2025.
- [32] Rasmus Palm, Ulrich Paquet, and Ole Winther. Recurrent relational networks. Advances in neural information processing systems, 31, 2018.
- [33] Subham S Sahoo, Marianne Arriola, Yair Schiff, Aaron Gokaslan, Edgar Marroquin, Justin T Chiu, Alexander Rush, and Volodymyr Kuleshov. Simple and effective masked diffusion language models. Advances in Neural Information Processing Systems, 37:130136–130184, 2024.
- [34] Shelly Sheynin, Adam Polyak, Uriel Singer, Yuval Kirstain, Amit Zohar, Oron Ashual, Devi Parikh, and Yaniv Taigman. Emu edit: Precise image editing via recognition and generation tasks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8871–8879, 2024.
- [35] Yuxuan Song, Zheng Zhang, Cheng Luo, Pengyang Gao, Fan Xia, Hao Luo, Zheng Li, Yuehang Yang, Hongli Yu, Xingwei Qu, et al. Seed diffusion: A large-scale diffusion language model with high-speed inference. arXiv preprint arXiv:2508.02193, 2025.
- [36] Mitchell Stern, William Chan, Jamie Kiros, and Jakob Uszkoreit. Insertion transformer: Flexible sequence generation via insertion operations. In International Conference on Machine Learning, pages 5976–5985. PMLR, 2019.
- [37] Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063, 2024.
- [38] Dimitri Von Rütte, Janis Fluri, Yuhui Ding, Antonio Orvieto, Bernhard Schölkopf, and Thomas Hofmann. Generalized interpolating discrete diffusion. arXiv preprint arXiv:2503.04482, 2025.
- [39] Guanghan Wang, Yair Schiff, Subham Sekhar Sahoo, and Volodymyr Kuleshov. Remasking discrete diffusion models with inference-time scaling. arXiv preprint arXiv:2503.00307, 2025.
- [40] Po-Wei Wang, Priya Donti, Bryan Wilder, and Zico Kolter. Satnet: Bridging deep learning and logical reasoning using a differentiable satisfiability solver. In International Conference on Machine Learning, pages 6545–6554. PMLR, 2019.
- [41] Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. Self-consistency improves chain of thought reasoning in language models. arXiv preprint arXiv:2203.11171, 2022.
- [42] Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli. Image quality assessment: from error visibility to structural similarity. IEEE transactions on image processing, 13(4): 600–612, 2004.
- [43] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824–24837, 2022.
- [44] Sean Welleck, Ilia Kulikov, Stephen Roller, Emily Dinan, Kyunghyun Cho, and Jason Weston. Neural text generation with unlikelihood training. arXiv preprint arXiv:1908.04319, 2019.
- [45] Cort J Willmott and Kenji Matsuura. Advantages of the mean absolute error (mae) over the root mean square error (rmse) in assessing average model performance. Climate research, 30(1): 79–82, 2005.
- [46] Zhihui Xie, Jiacheng Ye, Lin Zheng, Jiahui Gao, Jingwei Dong, Zirui Wu, Xueliang Zhao, Shansan Gong, Xin Jiang, Zhenguo Li, et al. Dream-coder 7b: An open diffusion language model for code. arXiv preprint arXiv:2509.01142, 2025.

- [47] Yi Xin, Qi Qin, Siqi Luo, Kaiwen Zhu, Juncheng Yan, Yan Tai, Jiayi Lei, Yuewen Cao, Keqi Wang, Yibin Wang, et al. Lumina-dimoo: An omni diffusion large language model for multi-modal generation and understanding. arXiv preprint arXiv:2510.06308, 2025.
- [48] Jiacheng Ye, Zhihui Xie, Lin Zheng, Jiahui Gao, Zirui Wu, Xin Jiang, Zhenguo Li, and Lingpeng Kong. Dream 7b: Diffusion large language models. arXiv preprint arXiv:2508.15487, 2025.
- [49] Yang Ye, Xianyi He, Zongjian Li, Bin Lin, Shenghai Yuan, Zhiyuan Yan, Bohan Hou, and Li Yuan. Imgedit: A unified image editing dataset and benchmark. arXiv preprint arXiv:2505.20275, 2025.
- [50] Kai Zhang, Lingbo Mo, Wenhu Chen, Huan Sun, and Yu Su. Magicbrush: A manually annotated dataset for instruction-guided image editing. Advances in Neural Information Processing Systems, 36:31428–31449, 2023.
- [51] Zhanhui Zhou, Lingjie Chen, Hanghang Tong, and Dawn Song. dllm: Simple diffusion language modeling. arXiv preprint arXiv:2602.22661, 2026.

### A Theoretical Analysis

#### A.1 Bayes-Consistent Revision-Policy Learning

Under a fixed, θ-independent corruption proposal, cross-entropy training on per-position oracle revision labels recovers the conditional distribution of the optimal revision action at every position (Theorem 1). Theorem 7 converts model-side TV distance into excess 0–1 risk; Proposition 9 shows the History Reference conditioning cannot increase Bayes risk.

Notation. Let V denote the token vocabulary, V¯ := V ∪ {MASK}, and E ⊆ {1,...,N} the set of edit positions. A training sample is drawn as follows. Sample (x∗,c) ∼ D, then a rule ρi ∼ πρ over {wrong,mask} independently at each i ∈ E. For ρi = wrong, draw a source token wi ∼ νwrong(· | x∗i ) on V \ {x∗i } and boundaries (βi,µi) from a fixed θ-independent law B on {1 ≤ β < µ ≤ T}; for ρi = mask, draw a single boundary µi ∼ Uniform{1,...,T}. The current-step index is t ∼ Uniform{0,...,T − 1}. Given these boundaries, the trajectory x˜(0:T) is per-position deterministic: wi → MASK → x∗i at (βi,µi) for wrong-rule positions, MASK → x∗i at µi for mask-rule positions, and x∗i throughout for non-edit positions i ∈/ E.

We write z := x˜(t) for the current state, H := x˜(<t) for the trajectory prefix, and ϕ(H,c) ∈ Rd for a deterministic history feature map (the history component of the HR accumulated embedding a(t); see Appendix B). The accumulated embedding satisfies a(t) = e(t) + ϕ(H,c) with e(t) = wte(z).

Let X := (z,ϕ(H,c),c,E) denote the analyst-side conditioning. Our proofs condition on X, while the implementation passes only (a(t),c) to the model and uses z,E to construct the supervision label. We write pθ(· | X)i for the model’s per-position categorical over V¯. By Proposition 9, the Bayes risk under σ(X) lower-bounds the Bayes risk achievable from σ(a(t),c), so Theorem 1’s population minimum is a lower bound on the implementation’s Bayes risk.

Oracle revision rule. For each i ∈ E, the deterministic oracle next-action is

 

MASK zi ∈ V \ {x∗i } (visible non-target token)

τ(zi,x∗i ) =

(5)

x∗i zi = MASK (reveal target) x∗i zi = x∗i (preserve target)



This is the single optimal revision action at the current state, not a per-step transition kernel for the trajectory.

Training objective. The training loss is the per-position three-state cross-entropy summed over edit positions:

−log pθ τ(zi,x∗i ) X i , (6) where D ⊗ q denotes the joint distribution generated by the sampling order above. Conditional label distribution. For each i ∈ E, define

Ltrain(θ) = ED⊗q

i∈E

τ(zi,x∗i ) = y . (7)

Pi∗(y | X) := Pr

(x∗,w,ρ,boundaries,t)∼(D⊗q)|X

This is the marginal distribution of the oracle label given the model’s input. Since X does not include x∗, Pi∗ is in general non-degenerate. Theorem 1 (Population minimizer recovers the conditional label distribution). Assume the rich-

family idealization: {pθ} is large enough that for some θ, the model’s per-position output pθ(· | X)i matches Pi∗(· | X) jointly at every X in the support of D ⊗ q and every i ∈ E. Then any population minimizer θ∗ ∈ arg minθ Ltrain(θ) satisfies

pθ∗(· | X)i = Pi∗(· | X) for (D ⊗ q)-a.e. X and every i ∈ E. Proof. Decompose Ltrain point-wise in X via the tower property: Ltrain(θ) = EX

EY

i|X[−log pθ(Yi | X)i] ,

i∈E

where Yi := τ(zi,x∗i ) has conditional distribution Pi∗(· | X). Per-position decoupling. At fixed X, the inner sum has terms depending on θ only through the per-position output distributions {pθ(· | X)i}i∈E, which the rich-family hypothesis allows θ to realize independently. Hence minimizing the sum jointly over θ is equivalent to minimizing each per-position term independently in its corresponding categorical. Per-position propriety. By strict propriety of cross-entropy [12], for any reference distribution r on V¯ the expected score Ey∼r[−log p(y)] is uniquely minimized over the categorical simplex at p = r. Apply this with r = Pi∗(· | X) to each position i; the unique per-position minimizer is pθ(· | X)i = Pi∗(· | X). The almost-everywhere conclusion follows because the support of D ⊗ q has full measure.

| |
|---|

- Remark 2 (well-specification). If {pθ} is not rich enough to represent Pi∗, the conclusion holds in the function-class limit: the minimizer is the closest representable distribution in expected KL.
- Remark 3 (loss reweighting). Theorem 1 is stated for the unweighted CE objective of Eq. (6). Adding any constant or X-measurable per-position weighting wi(X) > 0 does not change the minimizer, since the per-X minimization scales by wi(X). Label-dependent weighting tilts the minimizer but is not used in our experiments.

The general form of Theorem 1’s minimizer simplifies sharply at each of the two state types. Define αi(X) := Pr(x∗i = zi | X), the conditional probability that the visible token equals the gold target. Proposition 4 (Visible-token minimizer is a binary calibrated correctness belief). At any X with zi ̸= MASK,

##### Pi∗(· | X) = αi(X) · δz

##### + (1 − αi(X)) · δMASK.

i

Proof. Conditional on zi ̸= MASK, the value zi is determined by X. The remaining randomness in τ(zi,x∗i ) given X is the indicator 1{zi = x∗i }, which equals 1 with probability αi(X) and 0 otherwise. By Eq. (5), τ = zi in the first case and τ = MASK in the second.

| |
|---|

Interpretation. At visible-token positions, the population minimizer encodes a calibrated belief about whether the visible token is the gold target. The inference rule mask iff pθ(MASK | X) > pθ(zi | X) is the Bayes-optimal binary decision based on this belief, since the two probabilities are the atoms of Pi∗.

Remark 5 (degenerate case). If 1{zi = x∗i } is X-measurable, then αi(X) ∈ {0,1} a.s. and Pi∗ collapses to δMASK or δz

. The main results do not require this.

i

Proposition 6 (Calibrated gold belief at MASK positions). At any X with zi = MASK,

Pi∗(y | X) = Pr

[x∗i = y].

(x∗,w)∼(D⊗q)|X

Proof. When zi = MASK, Eq. (5) gives τ(zi,x∗i ) = x∗i tautologically, so the conditional distribution of τ given X equals the conditional distribution of x∗i given X.

| |
|---|

Interpretation. At MASK positions the population minimizer becomes a conditional language model over the gold token — the same target as [31]’s two-state model. Sampling from pθ(· | MASK,ϕ(H,c),c) recovers the standard mask-prediction sampling step.

Summary. RM trains a single per-position categorical that carries two calibrated beliefs: a binary {zi,MASK} correctness belief at visible-token positions, and a full-vocabulary gold-token belief at MASK positions. The inference rule reads each off the appropriate atoms — the re-mask check at visible positions and the candidate reveal at MASK positions in Eq. 1.

- A.2 Plug-in Excess-Risk Theorem We analyze the deterministic argmax plug-in policy

pθ(v | X)i,

aˆθ(X)i := arg max

v∈V¯

with ties broken by a fixed model-independent rule (e.g., lexicographic order on V¯); the Bayes-optimal classifier aˆ∗(X)i := arg maxv Pi∗(v | X) is defined under the same tie-breaking rule. Bounds below

hold for any such fixed choice. The per-position 0–1 revision risk against the oracle is

1 |E| i∈E

1[ˆaθ(X)i ̸= τ(zi,x∗i )] ,

R(θ) := E(D⊗q)

and the Bayes risk is R∗ := infp R(p). By Propositions 4–6, R∗ has irreducible contributions from both state types: 1 − maxy Pi∗(y | X) at MASK positions, and min(αi(X),1 − αi(X)) at visible-token positions. Both vanish only at degenerate X where Pi∗ is a delta.

Theorem 7 (Plug-in excess Bayes risk). Under the setup above,

R(θ) − R∗ ≤ 2 · EX

1 |E| i∈E

TV(pθ(· | X)i, Pi∗(· | X)) .

Proof. Fix X and i ∈ E; drop the i subscript on P∗. By definition Y := τ(zi,x∗i ) has conditional distribution P∗(· | X), so PrY ∼P∗[ˆa ̸= Y | X] = 1 − P∗(ˆa | X) for any deterministic classifier aˆ measurable in X.

Case aˆθ = aˆ∗. Per-X excess risk is zero; the bound is trivial. Case aˆθ ̸= aˆ∗. Per-X excess risk equals P∗(ˆa∗ | X) − P∗(ˆaθ | X). Decompose:

P∗(ˆa∗) − P∗(ˆaθ) = [P∗(ˆa∗) − pθ(ˆa∗)]

+[pθ(ˆa∗) − pθ(ˆaθ)]

+[pθ(ˆaθ) − P∗(ˆaθ)]

.

(A)

(B)

(C)

(B) ≤ 0 since aˆθ maximizes pθ (under the fixed model-independent tie-break stated above). Bound each of (A) and (C) by absolute value: (A) ≤ |pθ(ˆa∗) − P∗(ˆa∗)| and (C) ≤ |pθ(ˆaθ) − P∗(ˆaθ)|. Since aˆ∗ ̸= aˆθ, these are two distinct nonnegative terms in the sum v |pθ(v) − P∗(v)|, hence (A) + (C) ≤ v |pθ(v) − P∗(v)| = 2 · TV(pθ, P∗). Take expectation over X and average over i.

| |
|---|

Remark 8 (on the factor of 2). This is the standard Devroye–Györfi–Lugosi excess-risk bound [9, Theorem 2.2]. The constant 2 is tight: with P∗ = δ1 and pθ = (1/2 − ϵ,1/2 + ϵ), excess risk equals 1 while TV → 1/2.

Combining Theorems 1 and 7. At the population minimum pθ = P∗, so R(θ∗) = R∗: the CE-loss minimizer is Bayes-optimal under 0–1 revision risk. The Bayes risk R∗ is non-zero by construction (Propositions 4–6); driving CE loss further does not — and should not — reduce R∗.

#### A.3 History Reference Information Bound

History Reference augments the conditioning: the model receives not just (z,c) but (z,ϕ(H,c),c). For a random variable (or tuple) W, write σ(W) for the σ-algebra generated by W; a decision rule aˆ is σ(W)-measurable exactly when aˆ can be written as a deterministic function of W.

Proposition 9 (History monotonicity). Let σbase := σ(z,c) and σHR := σ(z,H,c). Then

E[1[ˆa ̸= τ]] ≤ inf

E[1[ˆa ̸= τ]].

inf

a σˆ HR-meas.

a σˆ base-meas.

Proof. σbase ⊆ σHR since (z,c) is a sub-tuple of (z,H,c). Hence every σbase-measurable estimator is also σHR-measurable, so the infimum over a larger family cannot exceed the infimum over a smaller one.

| |
|---|

The same argument applies to any history feature ϕ used by the model: σ(z,ϕ(H,c),c) ⊆ σ(z,H,c), so ϕ achieves a Bayes risk between the no-history baseline and the full trajectory. Whether the HR accumulated embedding strictly decreases Bayes risk relative to the no-history baseline is an empirical question answered in Section 4.

Information hierarchy. The data-processing argument gives a three-level hierarchy — current state only, current state plus the accumulated embedding, current state plus the full trajectory:

σ(z,c) ⊆ σ(z,ϕ(H,c),c) ⊆ σ(z,H,c). The history feature ϕ(H,c) is a compressed history signal; distinct trajectories generally produce distinct ϕ(H,c), but the mapping is many-to-one.

#### A.4 Top-k Variant: Distribution-Shift TV Bound

Theorem 1 assumes a fixed corruption ν. In practice we use a non-uniform θ-independent corruption ν(prac): a frozen pre-trained checkpoint’s top-k distribution for text generation, a fixed wrong-digit distribution for Sudoku, and the source-image VQ tokens for image editing. In the image-edit case the VQ token is taken as wi only when wi ̸= x∗i ; positions where the source already matches the target fall under the keep branch, keeping νwrong(prac) supported on V \ {x∗i }. The loss gap between νunif and ν(prac) decomposes in three layers.

Assumption 10 (Bounded log-likelihood). Per-token log-probabilities are clipped: −log pθ(v | X)i ≤ B := −log ε for every token v, position i, and input X. We use ε = 10−8, giving B ≈ 18.

- Layer 1 (joint TV bound). For any test function f with |f| ≤ B · |E| and any two probability measures P,Q on the joint sample space,

EPf − EQf ≤ 2B · |E| · TV(P,Q).

Applied to f = i∈E −log pθ(τ(zi,x∗i ) | X)i, with q(unif) and q(prac) the joint proposals under νunif and ν(prac) respectively:

L(unif)train (θ) − L(prac)train (θ) ≤ 2B · |E| · E(x∗,c)∼D TV q(unif)(· | x∗,c), q(prac)(· | x∗,c) .

- Layer 2 (factorize the joint TV). The two proposals share the rule prior πρ, the boundary law B, and the timestep distribution; they differ only in the conditional corruption at ρ = wrong positions.

Define the rule-marginalized corruption ν˜i(· | x∗i ) := πwrong · νwrong(· | x∗i ) + πmask · δMASK. The proposals agree on the πmask branch, so by the mixture-TV identity TV(αP1 + (1 − α)P2, αQ1 + (1 − α)P2) = α TV(P1,Q1), the per-position TV between rule-marginalized corruptions equals

πwrong · TV(νwrong(unif),i,νwrong(prac),i). Combining with the standard product-distribution TV inequality (independence across positions),

TV q(unif), q(prac) ≤ πwrong ·

i∈E

TV νwrong(unif),i, νwrong(prac),i .

- Layer 3 (per-position bound, ready for analysis). Combining Layers 1 and 2:

L(unif)train (θ) − L(prac)train (θ) ≤ 2B · |E| · πwrong · E(x∗,c)∼D

TV νwrong(unif),i, νwrong(prac),i . (8)

i∈E

Only "wrong"-rule positions contribute; the mask-rule and non-edit branches contribute zero.

Implication. Theorem 1 still applies on the support of q(prac): the population minimizer recovers the conditional label distribution under ν(prac) rather than under νunif. Since inference-time errors are not uniform, ν(prac) is the more relevant covered region.

### B Engineering Implementation

The relative-lag form of §3.2 admits a strict O(1) per-step update of the accumulated embedding a(it) (also written acci).

Derivation of the recurrence. Starting from the closed form (Eq. (2))

t

γ t−k Rk−t e(ik), e(ik) := wte(˜x(ik)), and the corresponding sum at step t − 1,

a(it) =

k=0

t−1

a(it−1) =

γ t−1−k Rk−(t−1) e(ik),

k=0

left-multiplying a(it−1) by γR−1 and using the rotation composition rule R−1Rk−(t−1) = Rk−t gives

t−1

γ R−1 a(it−1) =

γ t−k Rk−t e(ik).

k=0

Adding the current-step contribution e(it) (whose lag from the current step is zero, so R0 = I) recovers a(it):

a(it) = e(it) + γ R−1 a(it−1). This is the recurrence used in the implementation.

Memory layout. We maintain a single running tensor a ∈ RN×d, where N is the sequence length and d the hidden width. The per-step update is one N × d block-diagonal rotation followed by an element-wise add; storing the full token-level trajectory would cost O(T × N × d), while the recurrence collapses this to a single O(N × d) buffer.

Numerical stability. Each R∆ is a block-diagonal orthogonal transform, so R−1 preserves norms: ∥R−1 a(it−1)∥ = ∥a(it−1)∥. Combined with the decay factor γ ∈ (0,1], the running sum’s norm grows at most linearly in t.

Contrast with an absolute-step formulation. A natural alternative assigns each historical token an absolute-step rotation in a fixed (step-zero) reference frame, tk=0 γt−kRk e(ik). This admits the same per-step recurrence but anchors the trajectory at the start of denoising rather than the current step. The relative-lag form of Eq. (2) instead anchors the phase at the model’s most recent input, and is what we use in all reported experiments.

Loss reweighting. Training launchers may attach per-position weights wi to the CE term. The math launcher uses constant weights; the Sudoku launcher emphasizes edit positions via an X-measurable edit-vs-non-edit weight. Both fall under the loss-reweighting remark following Theorem 1, so the population minimizer is unchanged.

Bounded-loss floor. We clip output probabilities at ε = 10−8, giving B = −log ε ≈ 18 in Assumption 10.

### C Training Algorithm

We give the complete per-sample procedure summarized in Section 3.3. Let T = 6 denote the trajectory length used in our experiments.

- 1. Sample data. Draw (x∗,c) ∼ D (target sequence and task condition).
- 2. Sample per-position rule. For each i ∈ E, draw ρi ∼ πρ over {wrong,mask} indepen-

dently. Non-edit positions i ∈/ E are deterministically labeled "keep"; their state is x∗i at every t.

- 3. Sample source token (wrong-rule positions only). For each i ∈ E with ρi = wrong,

sample wi ∼ νwrong(· | x∗i ) on V \ {x∗i }. The rigorous baseline of Appendix A.1 uses uniform sampling; the practical variant (Appendix A.4) uses a task-specific non-uniform

ν(prac). Mask-rule positions (ρi = mask) need no source-token sample; they enter the trajectory directly at the MASK state.

- 4. Sample boundaries. For each ρi = wrong position, draw βi ∼ Uniform{1,...,T − 1}, then µi | βi ∼ Uniform{βi + 1,...,T}. For each ρi = mask position, draw µi ∼ Uniform{1,...,T}.
- 5. Construct the trajectory x˜(0:T). Per-position deterministic from the boundaries:

- • "wrong" rule: x˜(it) = wi for 0 ≤ t < βi, MASK for βi ≤ t < µi, x∗i for µi ≤ t ≤ T.
- • "mask" rule: x˜(it) = MASK for 0 ≤ t < µi, x∗i for µi ≤ t ≤ T.

- • "keep" / non-edit: x˜(it) = x∗i for all t.

The wrong-rule ordering wi → MASK → x∗i and the mask-rule ordering MASK → x∗i both hold by construction; no post-processing is required.

- 6. Sample current step. Draw t ∼ Uniform{0,...,T − 1} and set z = x˜(t), H = x˜(<t).
- 7. Construct labels. For each i ∈ E, the per-position label is the oracle action yi = τ(zi,x∗i ) defined in Eq. (5): MASK if zi is a visible non-target token, x∗i if zi = MASK (reveal target), x∗i if zi = x∗i (preserve target).
- 8. CFG dropout (image editing only). Draw text_drop ∼ Bernoulli(0.10) and image_drop ∼ Bernoulli(0.10) independently. If text_drop, replace the user instruction with its unconditional prefix. If image_drop, mark image-content positions to have their accumulated embedding overwritten by wte(MASK) after RMSNorm in step 9. The generic algorithm (text generation, Sudoku) skips this step.
- 9. Compute the accumulated embedding. Accumulate via Eq. (2), acci = tk=0 γ t−k Rk−t wte(˜x(ik)),

or by the recurrence acc(is) = wte(˜x(is)) + γR−1acc(is−1) with acc(0)i = wte(˜x(0)i ). Apply RMSNorm. For image editing, if image_drop from step 8 fires, overwrite acc at image-

content positions with wte(MASK). The model’s input is (acc,c); the pipeline keeps z aside for label construction and for the analyst-side conditioning of Theorem 1.

- 10. Compute the loss. Cross-entropy of Eq. (4) against the per-position labels yi from step 7.

Practical top-k proposal. For text generation, νwrong(prac)(· | x∗i ) in step 3 is the truncated softmax of a frozen checkpoint θ0’s top-k output (default k = 50), restricted to V \ {x∗i }; sharper conditionals admit smaller k. We probe θ0 once per training sample on x∗ and cache the per-position top-k index sets, drawing each position’s wrong token from its own conditional softmax. Since θ0 is fixed, the proposal is independent of the live parameter θ. Appendix A.4 bounds the objective shift relative to the uniform baseline.

Auxiliary losses (image editing only). In addition to the primary cross-entropy loss of Eq. (4), the image-editing pipeline uses an unlikelihood loss on non-edit positions [44] and a stage-2 ordering loss; their detailed form is deferred to a separate technical companion. These are heuristic regularizers outside the scope of Theorem 1.

### D Results gallery on image editing

- Fig. 5 provides additional image editing examples spanning object replacement, attribute modification, object insertion, and localized scene editing. Each row shows the source image, the predicted mask, our edit, and Lumina / Lumina-SFT baselines. The brighter regions in pixel-wise difference heat maps indicates larger pixel change relative to the source. Across these examples, our edits concentrate within the predicted-mask region; the baselines exhibit pixel changes outside that region (visible as off-mask hot regions in the heat maps). For readability, figures use abbreviated editing instructions; Tab. 5 maps these prompts to the full prompts.

E More results on text reasoning task

- Fig. 6 shows additional text revisions from our experiments. Cases 1 and 4: the model re-masks a token that is logically inconsistent with its surrounding context and re-predicts it. Cases 2 and

- 5: a single corrected token cascades downstream. After the earlier position is revised, the model also re-masks dependent positions further to the right and re-predicts them to be consistent with the corrected context. Case 3: format-level correction, where re-masking targets tokens whose surface form violates the answer template.

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

Add a Christmas tree in the left …

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

Add lace tablecloth cover the table …

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

Add a cluster of beans …

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

Replace with a decorative plate …

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

Turn teapot … into silver …

Original Predict Mask(Ours) Ours Lumina Lumina(SFT)

Turn fish into golden …

Turn table into wooden …

Turn muscle car into black …

Turn trees on the left into orange …

Add a ship …

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

Original

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

Predict Mask(Ours)

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

Ours

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

Ours heat map vs. Original

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

Lumina

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

Lumina heat map vs. Original

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

Lumina(SFT)

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

Lumina(SFT) heat map vs. Original

#### Figure 5: Additional qualitative results on image editing.

Table 5: Correspondence between the abbreviated editing instructions shown in the paper and the full editing instructions used in our experiments.

Abbreviated Editing Instruction Full Editing Instruction Turn time into black ... Turn time positioned in the right area into black. Add a person in wedding gown ... Add a person in an elaborate white wedding gown with lace, veil,

and tiara in the upper-middle of the image, occupying roughly half the height and most of the width.

Add a cabin made from wood ... Add a modern, minimalist cabin made from light-colored wood with a single door glowing orange, located in the upper-right portion and taking up nearly a quarter of the image.

Turn cookies into chocolate ... Turn cake positioned in the central area into chocolate. Turn trees on the left into orange ... Turn trees positioned on the left side into autumnal orange. Add bell-shaped blooms Add delicate, white bell-shaped blooms with slender green stems

in the middle-lower portion of the image, occupying a large area spanning most of the width.

Add cream piled on the top ... Add whipped cream piled on top of the French toast stack near the center-upper area of the image, covering about one sixth of the total image area.

Add creamy ... filling a spoon ... Add a white, creamy substance, likely coconut oil, filling a spoon and partially spilling onto a jar in the center-right-middle of the image, occupying about one-fifth of the area.

Turn the tree into snowy ... Turn tree positioned in the right area into snowy. Add a Christmas tree in the left ... Add a beautifully decorated Christmas tree covered with red, gold,

and silver ornaments and sparkling white lights in the left foreground, occupying most of the left and lower-central area.

Add lace tablecloth cover the table ... Add a small, round table covered with a lace tablecloth in the lowerleft-middle of the image, occupying about one fourth of the area.

Add a cluster of beans ... Add a cluster of ripening beans in various colors from green to red in the center-right of the image, occupying approximately one fourth of the area.

Replace with a decorative plate ... Add a decorative white plate on a wooden table in the bottom-middle

of the image, occupying about one third of the area. Turn teapot into silver ... Turn teapot positioned in the upper-central area into silver. Add a ship ... Add a large, fully-rigged sailing ship with golden sails and intricate

rigging near the center-upper-right, occupying about one third of the image area.

Turn muscle car into black ... Turn muscle car positioned in the central area into black. Turn fish into golden ... Turn fish positioned towards the lower central area into golden. Turn table into wooden ... Turn table positioned in the lower area into wooden.

[Figure 110]

#### Figure 6: Additional qualitative results on text generation.

