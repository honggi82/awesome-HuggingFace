## LangForce: Bayesian Decomposition of Vision Language Action Models via Latent Action Queries

Shijie Lian*12 Bin Yu*24 Xiaopeng Lin*35 Laurence Tianruo Yang61 Zhaolong Shen27 Changti Wu28 Yuzhuo Miao24 Cong Huang23 Kai Chen239

# arXiv:2601.15197v7[cs.AI]29May2026

### Abstract

Vision-Language-Action (VLA) models have shown promise in robot manipulation but often struggle to generalize to new instructions or complex multi-task scenarios. We identify a critical pathology in current training paradigms where goal-driven data collection creates a dataset bias. In such datasets, language instructions are highly predictable from visual observations alone, causing the conditional mutual information between instructions and actions to vanish, a phenomenon we term Information Collapse. Consequently, models degenerate into vision-only policies that ignore language constraints. To address this, we propose LangForce, enforces instruction following via Bayesian decomposition. By introducing learnable Latent Action Queries, we construct a dual-branch architecture to estimate both a visiononly prior p(a | v) and a language-conditioned posterior π(a | v,ℓ). We then optimize the policy to maximize the conditional Pointwise Mutual Information (PMI) between actions and instructions. This objective effectively penalizes the vision shortcut and rewards actions that explicitly explain the language command. Extensive experiments across on three benchmarks demonstrate substantial gains, including an 11.3% improvement on the challenging OOD SimplerEnv benchmark, validating the ability of LangForce to robustly ground language in action. Code and videos are available at this.

Work done at Beijing Zhongguancun Academy.

*Equal contribution. 1Huazhong University of Science and Technology 2Beijing Zhongguancun Academy 3Zhongguancun Institute of Artificial Intelligence 4Harbin Institute of Technology 5The Hong Kong University of Science and Technology (Guangzhou) 6Zhengzhou University 7Beihang University 8East China Normal University 9DeepCybot Co., Ltd.. Correspondence to: Laurence Tianruo Yang <ltyang@ieee.org>, Kai Chen <kaichen@zgci.ac.cn>.

Proceedings of the 43rd International Conference on Machine Learning, Seoul, South Korea. PMLR 306, 2026. Copyright 2026 by the author(s).

### 1. Introduction

Vision-Language-Action (VLA) models (Kim et al., 2024; Liu et al., 2025; Bjorck et al., 2025b; Black et al., 2025) have emerged as a promising paradigm for general-purpose robot manipulation, leveraging the vast knowledge of pretrained Vision-Language Models (VLMs) to ground natural language instructions into physical actions. By training on large-scale datasets of human demonstrations, these models aim to learn a policy π(a | v,ℓ) that can execute diverse tasks specified by language ℓ given visual observations v.

While demonstrating strong performance in in-distribution settings, current VLA models still face challenges in generalizing to novel instructions or complex multi-task scenarios, particularly in out-of-distribution (OOD) environments (Xu et al., 2025; Xing et al., 2025; Xu et al., 2023). This limitation is especially pronounced during post-training, where fine-tuning on narrow, task-specific datasets can lead to catastrophic forgetting of the VLM’s general capabilities and impair its ability to generalize to new tasks. We hypothesize that this fragility is exacerbated by a prevalent bias in current robotic datasets. Most robotic datasets are collected in a goal-driven manner, where a human operator performs a specific task repeatedly in a fixed scene. In such datasets, the mapping from visual scene v to language instruction ℓ is nearly injective: seeing a cabinet in the scene almost invariably implies the task “open the cabinet,” while seeing a bottle implies “pick up the bottle.” This deterministic coupling results in a sharp conditional distribution p(ℓ | v).

From a Bayesian perspective, the optimal policy can be decomposed as:

p(ℓ | a,v)p(a | v) p(ℓ | v)

. (1)

π(a | v,ℓ) =

Here, p(a | v) represents a vision-only prior (i.e., what actions are likely in this scene?), and p(ℓ | a,v) is the likelihood (i.e., how well does action a explain instruction ℓ?). When p(ℓ | v) is sharp, the model can predict ℓ solely from v without attending to a. Consequently, the likelihood term p(ℓ | a,v) collapses to p(ℓ | v), and the posterior policy degenerates to the prior:

π(a | v,ℓ) ≈ p(a | v). (2)

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

###### (a) Examples from Train Data (b) Inference on the 'PnPCanToDrawerClose’ Task in RoboCasa without Language Instruction

- Figure 1. Examples of the vision shortcut in RoboCasa (Nasiriany et al., 2024). Training data exhibits visual diversity but limited task diversity. As a result, the model learns to execute tasks directly based on specific visual cues rather than relying on language instructions.

In other words, the model effectively ignores the language instruction, learning a “vision shortcut” that fails whenever the task is ambiguous or the environment changes.

policy p(a | v) rather than a true language-conditioned policy π(a | v,ℓ). Specifically, we employ the Qwen3VL4B-GR00T model from starVLA (Community, 2026) as our representative VLA architecture. We conduct three pilot experiments to reveal this illusion of instruction following. In all three experiments, we train the model by feeding only the visual observation v (masking the language instruction ℓ), effectively testing the vision-only prior p(a | v).

To address this, we propose LangForce, a novel framework that explicitly enforces instruction following via Bayesian decomposition. Our key insight is to maximize the conditional Pointwise Mutual Information (PMI) between actions and instructions, which is equivalent to maximizing the loglikelihood ratio (LLR): log p(ℓ | a,v) − log p(ℓ | v). This objective penalizes the vision shortcut by requiring the action a to provide additional information about ℓ that cannot be inferred from v alone.

#### 2.1. Experiment 1: The Vision Shortcut in ID Testing

We first train a standard VLA model on a subset of the Humanoid robot tabletop manipulation data from PhysicalAIRobotics-GR00T-X-Embodiment-Sim (Bjorck et al., 2025b) and evaluate on 24 tasks from the RoboCasa benchmark (Nasiriany et al., 2024). Averaged across all 24 tasks, the vision-only model achieves a success rate of 44.6%, which is close to the language-conditioned baseline of 47.8% (as shown in Table 2). This small gap reveals that the model can succeed without relying on language instructions, as the training and evaluation scenes and tasks are highly similar, enabling the model to learn a near-deterministic mapping from vision to action. Figure 1 provides a relevant example.

We instantiate this framework by introducing Latent Action Queries—a set of learnable tokens injected into the VLM. These queries serve a dual purpose: they act as a bottleneck to extract action-relevant features for a downstream Diffusion Transformer (DiT) policy, and they enable a dualbranch training strategy. In the Priori Branch, queries attend only to vision to learn p(a | v); in the Posteriori Branch, they attend to both vision and language to learn π(a | v,ℓ). By optimizing the LLR between these branches, LangForce learns to ground language robustly without requiring new data. Our contributions are three-fold:

#### 2.2. Experiment 2: Failure in Ambiguous Scenarios

- 1. We identify and empirically validate the “vision shortcut” pathology in current VLA training, showing that standard models often ignore language in favor of dataset-specific visual correlations.
- 2. We propose LangForce, leveraging Latent Action Queries and a dual-branch Bayesian objective to recover language-conditioned policies from biased data.
- 3. We demonstrate that LangForce achieves state-of-theart performance on SimplerEnv and RoboCasa, with a remarkable 8.8% improvement in OOD generalization on SimplerEnv, proving its effectiveness in breaking the vision shortcut.

### 2. Motivation: Vision Shortcut

Before detailing our method, we present empirical evidence to substantiate our hypothesis: that standard VLA models trained on goal-driven datasets often learn a vision-only

To further investigate this behavior, we train a standard VLA model on the LIBERO benchmark (Liu et al., 2023), which contains four subsets: Spatial, Object, Long, and Goal. We train on all four training sets and evaluate on all four test sets. The vision-only model achieves success rates comparable to the full VLA model on three subsets (Spatial: 90.2%, Object: 99.6%, Long: 86.0% in VisionOnly, Spatial: 97.8%, Object: 98.8%, Long: 92.0% in Baseline), where each visual scene corresponds to a single task. However, on the LIBERO Goal subset, the vision-only success rate plummets to 9.8% (97.4% in Baseline).

The key difference is that LIBERO Goal presents inherent ambiguity: multiple valid tasks are associated with the same object configuration during training. For instance, a scene with multiple bowls, a stove, and a drawer could correspond to either “put bowl in drawer” or “put bowl on stove”. This confirms that while the model can exploit vision-action correlations in unambiguous datasets, when multiple tasks share the same visual context, due to a lack of language to

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

0.8

Vision & Language

0.7 Vision Only

0.6

ActionLoss

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

0.5

0.4

0.3 0.2

###### …

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

0.1

0 10000 20000 30000 40000

Step

(a) In LIBERO Goal, the Same Scene Corresponds to Multiple Tasks. (b) Action Loss in BridgeDataV2 + Fractal

- Figure 2. (a) In LIBERO Goal (Liu et al., 2023), the same scene corresponds to multiple tasks, revealing ambiguity that vision-only models fail to resolve. (b) Action loss curves on BridgeDataV2 (Walke et al., 2023) and Fractal (Brohan et al., 2022) show the vision-only model achieves comparable loss to the full model, indicating the presence of visual shortcuts even in diverse, in-the-wild datasets.

resolve ambiguity, the model is dominated by the prior p(a | v) learned from dataset statistics. Figure 2(a) illustrates examples where the same visual scene in LIBERO Goal corresponds to multiple distinct tasks.

I(ℓ;a | v), meaning the action choice significantly reduces uncertainty about the instruction. However, the CMI is bounded by the conditional entropy of the language:

I(ℓ;a | v) = H(ℓ | v) − H(ℓ | a,v) ≤ H(ℓ | v). (3)

#### 2.3. Experiment 3: Failure in OOD Generalization

In goal-driven datasets, the deterministic mapping v → ℓ implies H(ℓ | v) ≈ 0 (Xu et al., 2025). Consequently, I(ℓ;a | v) is forced to vanish, theoretically preventing the model from learning any dependency between a and ℓ beyond what is already captured by v. To break this deadlock, we cannot rely on standard imitation learning. Instead, we must explicitly intervene to maximize the information gain provided by the action. This motivates our use of the LogLikelihood Ratio (LLR), which effectively estimates the Pointwise Mutual Information (PMI), rewarding the policy only when it captures the specific semantics of ℓ that are not predictable from v.

Finally, we test the generalization capability by training on the high-quality BridgeDataV2 dataset (Walke et al., 2023) and Fractal (Brohan et al., 2022) dataset (diverse, in-thewild scenes) and evaluating on SimplerEnv (Li et al., 2024b) (simulation, OOD). During training on BridgeDataV2 and Fractal, the vision-only model achieves an action loss of 0.13, comparable to the full language-conditioned model’s loss of 0.08 (as shown in Figure 2(b)). This indicates that even in diverse, in-the-wild scenarios, the model can still identify visual shortcuts (e.g., specific lighting or background features mapping to specific actions) to minimize the training objective without truly grounding the language instructions. This open-loop loss trend serves as an optimization-level indication that vision-only shortcuts can fit the training distribution reasonably well.

### 3. Method: LangForce

In this section, we introduce LangForce, a framework designed to mitigate the vision shortcut in VLA models. We present the overall framework of LangForce in Figure 3. We first formalize the problem through a Bayesian lens (Section 3.1), deriving an objective that maximizes the mutual information between actions and instructions. We then present our architecture, which uses Latent Action Queries to instantiate this decomposition (Section 3.2), and detail our dual-branch training strategy (Section 3.3).

The policy-level evidence comes from downstream execution. When evaluated on SimplerEnv, which presents visually distinct simulation environments, the vision-only baseline achieves near 0% success despite fitting the training distribution reasonably well. This is the key evidence for the vision shortcut: it can produce seemingly reasonable optimization behavior on the training domain, while failing catastrophically when the policy is executed out of distribution.

#### 3.1. Objective Formulation

#### 2.4. Theoretical Insight: Information Collapse

We formalize the “vision shortcut” as a collapse of the conditional mutual information (CMI) between instructions and actions. Ideally, a robust VLA policy should maintain high

As established in Section 2.4, standard VLA training on goal-driven datasets leads to information collapse where π(a | v,ℓ) → p(a | v). To counteract this, we propose to regularize the policy by maximizing the conditional Pointwise Mutual Information (PMI) between the action and the

continuous actions

|ℒLLR defined in Eq (7)| |
|---|---|
| | |

condition

denoise

LLM

###### Action Expert

Priori Branch Posteriori Branch

Vision Token 𝑣 Action Query 𝒬 Language Token 𝓁

- Figure 3. The framework of LangForce. The framework employs a dual-branch architecture with shared VLM weights. The Priori Branch (left) processes [v, Q, ℓ] with causal masking to learn the vision-only prior p(a | v). The Posteriori Branch (right) processes [v, ℓ, Q] to learn the full policy π(a | v, ℓ). Latent Action Queries Q serve as a bottleneck interface, and the LLR objective (in Eq. 7) encourages the model to maximize the information between actions and instructions. At inference, only the Posteriori Branch is used, incurring no additional computational overhead.

instruction. This objective can be formulated as maximizing the Log-Likelihood Ratio (LLR) between the posterior policy and the vision-only prior:

π(a | v,ℓ) p(a | v)

= log p(ℓ | a,v) − log p(ℓ | v).

LLLR = log

(4) The detailed derivation is provided in Appendix A. This formulation requires us to simultaneously model the posterior π(a | v,ℓ) and the prior p(a | v). In the following sections, we describe how LangForce efficiently instantiates these two distributions using a shared architecture with Latent Action Queries.

#### 3.2. Latent Action Queries

To efficiently instantiate the proposed Bayesian decomposition within a unified VLM architecture, we introduce Latent Action Queries. We extend the VLM vocabulary with K = 64 learnable tokens, denoted as Q = {<|action 1|>,...,<|action K|>}. Correspondingly, the VLM’s embedding layer is expanded with K learnable embedding vectors. These vectors are learned to aggregate task-relevant information from the preceding vision and language tokens into a compact latent representation for action execution.

These queries function as a dedicated bottleneck at the interface between the VLM (e.g., Qwen3-VL (Bai et al., 2025)) and the continuous action head (a Diffusion Transformer (Peebles & Xie, 2023)). Crucially, while the VLM typically processes the full sequence, the bottleneck is enforced by exclusively forwarding the hidden states of these query tokens, HQ ∈ RK×D, to the action expert. This contrasts with recent VLA architectures such as π0 (Black et al., 2024) and GR00T (Bjorck et al., 2025b; GEAR-Team et al., 2025),

which typically feed the hidden states of all input tokens to the action head.

This design choice is critical: by leveraging the causal masking inherent in decoder-only VLMs, we can precisely control the information encoded in HQ simply by changing the position of Q in the input sequence. This flexibility enables the strict separation of vision-only and vision-language contexts required for our dual-branch strategy.

#### 3.3. Dual-Branch Training Framework

We propose a training paradigm with two parallel branches sharing the same VLM weights but different input.

- 1. Priori Branch (Vision-Only). To estimate the prior p(a | v), we construct the input sequence as:

Inputprior = [v,Q,ℓ]. (5) Due to the causal attention mask of the decoder-only VLM, the tokens in Q can attend to the visual observation v but cannot attend to the language instruction ℓ (which appears later). Thus, the hidden states HpriorQ encode purely visual information. We detach HpriorQ from the computation graph when optimizing Lprior to ensure that the gradient updates for the vision-only prior are confined to the DiT action head, preventing the shared VLM backbone from learning visual shortcuts. We use these features to predict the action a, optimizing a flow-matching loss Lprior to learn the dataset’s inherent action bias.

- 2. Posteriori Branch (Vision + Language). To estimate the true policy π(a | v,ℓ), we arrange the input as:

Inputpost = [v,ℓ,Q]. (6) Here, Q appears after ℓ, allowing it to attend to both vision and language. The resulting hidden states HpostQ encode the

full context. We optimize a main flow-matching loss Lmain to learn the expert action.

- 3. Maximizing the Likelihood Ratio. In addition to action prediction, we explicitly optimize the LLR objective. We treat the VLM’s language modeling loss as a proxy for log p(ℓ | ...). Specifically, in the Priori Branch, the language tokens ℓ attend to [v,Q]. Since Q encodes the action information a (via the prior), the probability of generating ℓ

in this branch approximates p(ℓ | v,aprior). In the Posteriori Branch, we can compute a baseline p(ℓ | v) by detaching gradients. However, a more direct and numerically stable approach is to maximize the difference in log-probabilities of the language tokens between the two branches. We define the LLR loss as:

LLLR = log p(ℓ | v,HpriorQ ) − sg(log p(ℓ | v)), (7)

where sg(·) denotes the stop-gradient operator. We maximize this term (minimize −LLLR) to force the action representations HQ to carry information that explains ℓ. The stop-gradient operation is employed to prevent the model from trivially maximizing the ratio by degrading the baseline p(ℓ | v) (i.e., damaging the VLM’s general language capabilities) rather than improving the numerator.

#### 3.4. Total Training Objective

We train the action decoder using the Rectified Flow Matching objective (Liu et al., 2022; Bjorck et al., 2025b). Specifically, we apply this objective to both the Priori Branch (conditioned on HpriorQ ) and the Posteriori Branch (conditioned on HpostQ ). Given a condition C ∈ {HpostQ ,HpriorQ }, the flow-matching loss is defined as:

0,a1 ||vψ(at,t,C) − (a1 − a0)||2 ,

LFM(ψ;C) = Et,a

(8) where vψ is the Diffusion Transformer (DiT) predicting the velocity field, a1 is the ground truth action trajectory, a0 ∼ N(0,I) is sampled from a standard Gaussian, and at = (1 − t)a0 + ta1 represents the interpolated state at timestep t ∈ [0,1].

The final training loss combines the action prediction losses from both branches with the LLR maximization term:

Ltotal = (1−λ)LFM(ψ;HpostQ )+λLFM(ψ;HpriorQ )−βLLLR,

(9) where λ balances the contribution of the prior and posterior action losses, and β controls the strength of the LLR regularization. We set λ = 0.3 and β = 0.1 in our experiments. During inference, we exclusively execute the Posteriori Branch to obtain HpostQ and generate actions via the DiT. This ensures that our method incurs no additional computational overhead compared to standard VLA baselines at test time.

### 4. Experiment

To comprehensively evaluate the effectiveness of LangForce, we conduct extensive experiments on two main simulation benchmarks, SimplerEnv and RoboCasa, with additional LIBERO results reported in Appendix B.1. We also evaluate LangForce in two real-world robot settings: colored-block pick-and-place and vegetable pick-and-place. Our training pipeline is built upon the StarVLA framework (Community, 2026), distributed across 8 NVIDIA H100 GPUs, and strictly follows its default training protocols to ensure fair comparison. In our experiments, LangForce is instantiated on the QwenGR00T architecture from StarVLA. We employ the AdamW optimizer (Loshchilov & Hutter, 2017) initialized with a learning rate of 1e-5 and a cosine annealing schedule. System-level optimizations include DeepSpeed ZeRO-2 (Rasley et al., 2020), gradient clipping at a norm of 1.0, and no gradient accumulation. All baseline performance metrics are obtained from their original papers or other peer-reviewed publications. To ensure a fair comparison, the training datasets for these baselines encompass the data used in our experiments.

#### 4.1. Experiments on SimplerEnv

We utilize two large-scale subsets from the Open XEmbodiment (OXE) dataset: BridgeDataV2 (Walke et al., 2023) and Fractal (Brohan et al., 2022). The model is finetuned for 50k steps on a cluster of 8 GPUs (batch size 16 per device). This benchmark includes four manipulation tasks: “Put spoon on towel”, “Put carrot on plate”, “Stack green cube on yellow cube”, and “Put eggplant in yellow basket”. For each task, we evaluate the VLA policies using the official evaluation scripts provided by the SimplerEnv repository (Li et al., 2024b). To mitigate the effects of randomness, we run 480 independent trials and report the average performance (Avg@480).

The results are summarized in Table 1. LangForce consistently outperforms comparison baselines, achieving a stateof-the-art average success rate of 66.5%. Notably, compared to the direct baseline QwenGR00T (55.2%) built on the same StarVLA framework, our method delivers an absolute improvement of 11.3%, validating that the performance gain stems from our proposed Bayesian decomposition rather than the base architecture. Significant improvements are observed in tasks requiring precise object identification and manipulation, such as “Put Carrot on Plate” (+13.6%) and

“Put Eggplant in Yellow Basket” (+15.0%). Furthermore, LangForce surpasses other recent strong competitors, including the flow-matching-based π0.5 (57.1%) and the dualsystem Isaac-GR00T-N1.6 (57.1%). These results confirm that by explicitly optimizing the mutual information between language and action, LangForce effectively mitigates

###### Table 1. Results of evaluating the VLA models with the WidowX robot in the SimplerEnv simulation environment. We highlight the best results in bold and the second-best results with underline.

Put Spoon on Towel

Put Carrot on Plate

Stack Green Block on Yellow Block

Put Eggplant in Yellow Basket

Method

Average

OpenVLA-OFT (Kim et al., 2025) 34.2 30.0 30.0 72.5 41.8 RoboVLM (Li et al., 2024c) 50.0 37.5 0.0 83.3 42.7 Magma (Yang et al., 2025a) 37.5 29.2 20.8 91.7 44.8 CogACT (Li et al., 2024a) 71.7 50.8 15.0 67.5 51.3 SpatialVLA (Qu et al., 2025) 20.8 20.8 25.0 70.8 34.4 TraceVLA (Zheng et al., 2025b) 12.5 16.6 16.6 65.0 27.7 VideoVLA (Shen et al., 2025) 75.0 20.8 45.8 70.8 53.1

π0 (Black et al., 2024) 29.2 62.5 29.2 91.6 53.1 π0.5 (Black et al., 2025) 49.3 64.7 44.7 69.7 57.1 Isaac-GR00T-N1.6-Bridge (GEAR-Team et al., 2025) 64.5 65.5 5.5 93.0 57.1

QwenGR00T (Baseline) + Qwen3-VL-4B (Community, 2026) 87.5 50.0 29.2 54.2 55.2 LangForce + Qwen3-VL-4B 89.6 63.8 33.3 79.2 66.5

###### Table 2. Results of evaluating the VLA models with the GR1 robot in the RoboCasa Tabletop simulation environment. The results for Isaac-GR00T N1.5 and Isaac-GR00T N1.6 are sourced from the official Isaac-GR00T github repository (Bjorck et al.,

- 2025b). The results for the first four baseline methods are sourced from the official starVLA experiments (Community, 2026). Performance on all 24 tasks can be found in Table 9 of the Appendix.

Method +Qwen3VLQwenFAST QwenGR00T+Qwen3VL +Qwen3VLQwenPI +Qwen3VLQwenOFT Average 39.0 47.8 43.9 48.8

Method Isaac-GR00TN1.5 Isaac-GR00TN1.6 QwenGR00TVisionOnly +Qwen3VLLangForce Average 48.2 47.6 44.7 52.6

the vision shortcut. Fundamentally, this validates that our approach prevents the policy from collapsing into a spurious vision-only prior p(a|v) caused by dataset determinism, and instead compels the model to learn the true causal dependency of actions on language instructions.

#### 4.2. Experiments on RoboCasa

We evaluate our method on the RoboCasa GR1 Tabletop Manipulation Benchmark (Nasiriany et al., 2024), which consists of 24 diverse manipulation tasks. These tasks feature complex interactions with articulated objects and varied geometries, exemplified by specific tasks like “PnPBottleToCabinetClose” and “PnPCanToDrawerClose”, as well as scenarios involving appliances like microwaves and toasters. For training, we utilize the Humanoid Robot Tabletop Manipulation subset from the PhysicalAI-Robotics-GR00TX-Embodiment-Sim (Bjorck et al., 2025b) dataset. All other settings follow Section 4.1. To guarantee statistical significance, we evaluate each task using 50 independent trials and report the average success rate (Avg@50).

The quantitative results on RoboCasa are presented in Ta-

ble 2. LangForce achieves a state-of-the-art average success rate of 52.6%, surpassing all competing baselines including QwenOFT (48.8%), Isaac-GR00T N1.5 (48.2%), and the direct baseline QwenGR00T (47.8%). Notably, our method substantially outperforms the VisionOnly baseline (44.7%), demonstrating the effectiveness of language grounding. Detailed per-task results can be found in Appendix B.2.

#### 4.3. Experiments on Real World

We evaluate LangForce on a real-world robotic setup using a Franka Research 3 arm. We design a “Pick and Place All Blocks” task involving colored blocks to assess instruction following and OOD generalization. We intentionally focus on pick-and-place settings because they provide a transparent testbed for separating instruction following from dexterous control: when there is only one plausible target in the scene, most VLA policies can succeed by relying primarily on manipulation competence, whereas selecting the correct target color or object category reveals whether the policy truly uses the language instruction. Notably, while the training demonstrations only contain single-block manipulation, we evaluate the model in scenarios requiring it to sequentially pick up multiple blocks (1, 2, or 3) and place them into the box. This setting serves as a rigorous test for the model’s generalization capabilities and long-horizon planning skills. Detailed experimental setup is provided in Appendix B.3.

As shown in Table 3, our model achieves a success rate of 25/30 on in-domain colors (single block), outperforming the QwenGR00T baseline (21/30). Consistent with our SimplerEnv findings (where tasks like “Stack Green Cube” requiring precise low-level control showed smaller gains), the improvement in the in-domain setting is moderate. This is expected, as simple block manipulation primarily challenges

- Table 3. Real World Pick and Place All Blocks experiment. Success rates (%) of the pick-and-place task on the Franka Research 3 robot. The headers 1, 2, and 3 denote the number of blocks in the scene that need to be placed into the box. We evaluate performance for both In-Domain and Out-of-Domain (unseen object colors).

Method

In-Domain Out-of-Domain 1 2 3 1 2

OpenVLA (Kim et al., 2025) 6/30 2/30 0/30 0/30 0/30 π0.5 (Black et al., 2025) 25/30 19/30 13/30 7/30 1/30 QwenGR00T 21/30 14/30 9/30 2/30 0/30 LangForce (Ours) 25/30 18/30 14/30 9/30 3/30

- Table 4. Real-world vegetable Pick and Place experiment. We use the same Franka Research 3 setup as the “Pick and Place All Blocks” task and evaluate each method over 30 trials per vegetable.

Vegetable LangForce (Ours) QwenGR00T Eggplant 24/30 (80.0%) 16/30 (53.3%) Pepper 25/30 (83.3%) 21/30 (70.0%) Carrot 27/30 (90.0%) 18/30 (60.0%) Cucumber 21/30 (70.0%) 16/30 (53.3%) Overall 97/120 (80.8%) 71/120 (59.2%)

the robot’s precise motor control rather than its high-level language understanding. However, on the challenging outof-distribution (OOD) task involving the unseen Red block, our model achieves a success rate of 9/30, significantly surpassing QwenGR00T, which struggles with a success rate of only 2/30. This contrast highlights that while LangForce maintains competitive low-level control capabilities, its primary advantage lies in enhanced instruction following and generalization to novel semantic concepts.

To further evaluate the method on objects with more diverse shapes and appearances, we additionally conduct a real-world “Vegetable Pick and Place” task using the same robot setup as the “Pick and Place All Blocks” task. We collect 100 demonstrations for each of four vegetables (eggplant, pepper, carrot, and cucumber) in the same scene, train LangForce and QwenGR00T under the same protocol, and evaluate each method over 30 trials per vegetable. As shown in Table 4, LangForce achieves an overall success rate of 80.8% (97/120), substantially outperforming QwenGR00T’s 59.2% (71/120). The consistent gains across all four objects indicate that the benefit of language-grounded action learning transfers beyond colored-block manipulation to real objects with varied geometry and appearance. Nevertheless, these real-world experiments remain limited to relatively simple pick-and-place behaviors and do not fully validate contact-rich or dexterous manipulation, which we leave for future work.

#### 4.4. Preservation of General Capabilities

A prevalent concern in the field is that fine-tuning VLMs for robotic action generation (VLA training) often compromises the model’s foundational reasoning and multimodal

Table 5. Ablation study on SimplerEnv. All experiments are based on the Qwen3-VL-4B backbone. We compare the baseline QwenGR00T, the addition of Latent Action Queries, and the full LangForce to validate the contributions of each component.

Put Spoon on Towel

Put Carrot on Plate

Stack Block

Eggplant in Basket

Method

Avg

QwenGR00T 87.5 50.0 29.2 54.2 55.2 + Action Query 74.6 58.3 29.2 67.9 57.5 LangForce 89.6 63.8 33.3 79.2 66.5

understanding, leading to a degradation of general conversational abilities (Zhou et al., 2025; Xu et al., 2025; Hancock et al., 2025; Yu et al., 2026b). ChatVLA (Zhou et al., 2025) attributes this to spurious forgetting, where robot training overwrites critical visual-text alignments, and task interference between control and understanding objectives.

We observe similar degradation in our baseline: as illustrated in Figure 4 and Figure 5, the standard QwenGR00T model loses the ability to converse coherently, even when prompted with pure text inputs. In stark contrast, LangForce remarkably preserves these capabilities when queried with language instructions. This is because our LLR objective enforces a strong dependency on language for action generation, preventing the linguistic representations from collapsing or becoming redundant. We provide a detailed discussion of this phenomenon and mechanisms in Appendix B.4.

#### 4.5. Ablation Studies

We conduct ablation studies on SimplerEnv to validate the contributions of individual components in LangForce. All experiments utilize the Qwen3-VL-4B backbone, and results are presented in Table 5.

Effectiveness of Bayesian Decomposition. Comparing the full LangForce (63.5%) with the “+ Action Query” ablation (57.5%), we observe a significant performance boost (+6.0%). This indicates that while the architectural changes provide some benefit, the core improvement stems from our dual-branch Bayesian learning objective. By explicitly modeling and maximizing the pointwise mutual information (PMI) between instructions and actions, the model effectively overcomes the vision shortcut, validating the central hypothesis of this work. Additionally, we provide ablation studies on the hyperparameters λ and β in Appendix B.5.

Potential of Latent Action Queries. Even without the dualbranch definition, introducing Latent Action Queries (“+ Action Query”) improves upon the QwenGR00T baseline (55.2% → 57.5%). This suggests that Latent Action Queries function as a promising architectural inductive bias. Unlike standard approaches that feed full sequences of vision and language token embeddings into the action decoder, our query-based mechanism forces the VLM to compress and summarize task-relevant information into a compact set

[Figure 24]

|If the line 𝑦 = 2 𝑥 + 5 is tangent to the curve 𝑦 = 𝑒𝑥 + 𝑥 + 𝑎, then 𝑎 = ?|
|---|

[Figure 25]

###### LLM in LangForce

We are given two functions:

Line: 𝑦 = 2𝑥 + 5 ; Curve: 𝑦 = 𝑒𝑥 + 𝑥 + 𝑎. We are told they are tangent to each other ⇒ they intersect at exactly one point AND have the same derivative there. Let’s denote the point of tangency as 𝑥 = 𝑡 . Then at this point:

[Figure 26]

[Figure 27]

###### Qwen3-VL-4B-Instruct

We are given that the line 𝑦 = 2𝑥 + 5 is tangent to the curve 𝑦 = 𝑒𝑥 +

𝑥 + 𝑎. That means: 1. The two graphs intersect at one point (since it's tangent). 2. At that point, their derivatives are equal (same slope).

- (1) The ordinates (y-values) are equal: 𝑒𝑡 + 𝑡 + 𝑎 = 2𝑡 + 5 (since both equal y)
- (2) The derivatives are equal: Derivative of curve: 𝑑𝑥𝑑 𝑒𝑥 + 𝑥 + 𝑎 = 𝑒𝑥 + 1. Derivative of line: 𝑑

𝑑𝑥 2𝑥 + 5 = 2. So:

- Step 1: Set the derivatives equal. The derivative of the line 𝑦 = 2𝑥 + 5 is 𝑑𝑦 𝑑𝑥 = 2 . The derivative of the curve 𝑦 = 𝑒𝑥 + 𝑥 + 𝑎 is 𝑑𝑦𝑑𝑥 =e𝑥 +1. Set them equal:

e𝑥+1 = 2 ⇒ e𝑥 = 1 ⇒ 𝑥 = 0 So the point of tangency occurs at 𝑥 = 0 .

- Step 2: Use the point of tangency to find 𝒂. At 𝑥 = 0 , the curve and the line must have the same 𝑦-value. Compute 𝑦 from the line at 𝑥 = 0 : 𝑦 =

𝑒𝑡 + 1 = 2 ⇒ 𝑒𝑡 = 1 ⇒ 𝑡 = 0

Now plug 𝑡 = 0 into equation. So, Left side: 𝑒0 + 0 + 𝑎 = 1 + 𝑎 . Right side: 2 0 + 5 = 5. So:

1 + 𝑎 = 5 ⇒ 𝑎 = 4 Final Answer: \𝒃𝒐𝒙𝒆𝒅{𝟒}

[Figure 28]

[Figure 29]

Loss of language ability

###### LLM in QwenGR00T

2 0 + 5 = 5. Compute 𝑦 from the curve at 𝑥 = 0 : 𝑦 = e0 +0 + 𝑎 = 5. Set them equal: 1 + 𝑎 = 5 ⇒ 𝑎 = 4

所有项一起微分的话，则: 业精进dentistate Dentistry Dentistry Dentistry Dentistry Dentistry Dentistry Dentistry Dentistry Dentistry Dentisty Dentistate Dentistate Dentistate Dentistate Dentistate Dentistate Dentistate Dentistate Dentistate Dentistate …

[Figure 30]

Final Answer: \𝒃𝒐𝒙𝒆𝒅{𝟒}

Figure 4. Qualitative comparison of general multimodal reasoning. We present a case where the model is asked to solve a mathematical problem. The standard VLA baseline (QwenGR00T) suffers from catastrophic forgetting; while the text before the comma implies “differentiating all terms together”, the subsequent output degenerates into repetitive and meaningless gibberish (bottom right). In contrast, LangForce (top right) retains the VLM’s original reasoning and language generation capabilities (left), successfully solving the problem.

of latent tokens. From a computational perspective, this design is highly efficient. It decouples the complexity of the Diffusion Transformer (DiT) from the length of the VLM input context. Specifically, the complexity of condition processing in the DiT is reduced from O(N2) (scaling with the massive number of vision-language tokens N) to O(K2) (scaling with the small, constant number of query tokens K), thereby streamlining the action generation process.

We also investigate the impact of the number of latent action queries on model performance. Detailed analysis and results are provided in Appendix B.6.

### 5. Related Work We build our work upon the following rigorous foundations: Vision-Language-Action Dataset and Benchmark.

The advancement of generalist robot policies relies heavily on large-scale datasets and rigorous benchmarks. LIBERO (Liu et al., 2023) pioneered the systematic study of knowledge transfer in lifelong robot learning. To scale up real-world data, BridgeData V2 (Walke et al., 2023) provided diverse interaction trajectories on low-cost hardware. This effort was expanded by Open X-Embodiment (OXE) (O’Neill et al., 2024), which aggregated data across 22 robot embodiments, and Droid (Khazatsky et al., 2024), which further increased diversity with distributed data collection. For scalable evaluation, RoboCasa (Nasiriany et al., 2024) introduced a large-scale simulation framework with realistic kitchen environments, while SimplerEnv (Li et al., 2024b) provided a simulated evaluation proxy to correlate with real-world performance, addressing the reproducibility crisis in physical evaluation. More recently, RoboTwin

2.0 (Chen et al., 2025) offered a unified benchmark for bimanual manipulation with automated data generation, and AgiBot-World (Bu et al., 2025) scaled training data to over 1 million trajectories with human-in-the-loop verification.

#### Vision-Language-Action Models.

To bridge the gap between semantic understanding and physical control, Vision-Language-Action (VLA) models have emerged as a dominant paradigm. Early works like OpenVLA (Kim et al., 2024) and its variant OpenVLA-OFT (Kim et al., 2025) fine-tune large language models for robotic control. Further architectural innovations include using diffusion transformers (Li et al., 2024a; Liu et al., 2025) or dual-system designs like the GR00T series (Bjorck et al., 2025b;a; GEAR-Team et al., 2025), which couples a VLM with a diffusion head. Recently, the π0 series (Black et al., 2024; 2025; Pertsch et al., 2025) utilizes data from multiple robots, high-level semantic prediction, web data, and other sources to enable broadly generalizable real-world robotic manipulation

Other approaches like X-VLA (Zheng et al., 2025a) introduce embodiment-specific soft prompts to facilitate crossembodiment generalization. By learning separate sets of embeddings for each data source, X-VLA effectively leverages heterogeneous robot data with minimal additional parameters. SpatialVLA (Qu et al., 2025) argues that spatial understanding is central to manipulation, introducing Ego3D Position Encoding and Adaptive Action Grids to inject 3D information and learn transferable spatial action knowledge. 3DMix (Yu et al., 2026a) further studies VGGT-based 3D feature integration and proposes a plug-and-play gated fusion module that adaptively balances 2D semantic and 3D geometric features for VLA models. Then, VideoVLA (Shen

et al., 2025) explores transforming video generation models into robot manipulators. By jointly predicting action sequences and future visual outcomes, it leverages the “visual imagination” of generative models to enhance generalization across novel tasks and objects. TwinBrainVLA (Yu et al.,

- 2026b) mitigates catastrophic forgetting by coordinating a frozen generalist “Left Brain” for semantic understanding and a trainable specialist “Right Brain” for sensorimotor learning, effectively balancing high-level reasoning with low-level control. BayesVLA (Xu et al., 2025) also employs Bayesian decomposition to improve instruction following. It utilizes a two-stage framework: first training a vision-conditioned prior on large-scale vision-action pairs, and then freezing this prior to train a language-conditioned posterior. However, this design strictly requires a separated training process where the vision component must be frozen in the second stage. In contrast, LangForce enables singlestage, end-to-end training.

Overall, other VLA approaches lack systematic solutions to the vision shortcut problem, where models ignore language instructions in goal-driven datasets.

### 6. Discussion

Our analysis in previous sections suggests that the deterministic mapping between visual observations and language instructions in goal-driven datasets may facilitate the vision shortcut. Consequently, we advocate for:

Prioritizing ambiguous scenarios during data collection to naturally compel models to rely on language for disambiguation.

This data-centric direction is a long-term and fundamental solution. At the same time, a large amount of existing VLA data has already been collected under approximately goal-deterministic settings, and recollecting or re-annotating such data at scale is expensive. LangForce provides a way to make better use of these existing datasets. Moreover, even with broader data, shortcut behavior can still arise whenever optimization exploits local scene-to-task correlations. An explicit objective that contrasts the language-conditioned posterior with a vision-only prior therefore remains valuable for encouraging genuine language use.

Additionally, for further insights on data collection, leveraging human data, and connections to World Models, we refer readers to the detailed discussion in Appendix C.

### 7. Limitation

While LangForce offers significant improvements in robustness, the dual-branch architecture introduces a limitation regarding computational overhead during training. Since the

model must compute both the Priori and Posteriori branches, the computational cost per iteration theoretically increases. However, we note that the visual input prefix is identical for both branches, and the number of visual tokens vastly outnumbers that of the language and latent action query tokens. By employing a prefix prefill strategy to compute and reuse the visual representations (e.g., vision encoder outputs) for both branches, the actual increase in training time is marginal. Thus, the additional computational overhead remains within a completely acceptable range.

Our real-world experiments are limited to relatively simple pick-and-place settings that primarily evaluate instruction following rather than dexterous manipulation. This aligns with the goal of improving language grounding rather than low-level manipulation skill. Using simple tasks makes failures easier to attribute to insufficient instruction following instead of execution errors.

### 8. Conclusion

In this work, we identify the vision shortcut in VLA training, where policies rely on visual correlations rather than language under approximately goal-deterministic data. We analyze this failure mode through a Bayesian and informationtheoretic lens and introduce LangForce, a post-training framework that contrasts a language-conditioned posterior with a vision-only prior to encourage instruction-specific action information. With Latent Action Queries and a shared dual-branch architecture, LangForce improves language grounding without inference overhead. Across SimplerEnv, RoboCasa, LIBERO, and two real-world pick-and-place settings, LangForce improves over VLA baselines in ambiguous and OOD scenarios. These results highlight the importance of objectives that prevent shortcuts and preserve the causal role of language in action generation. We hope this perspective encourages future VLA research to evaluate not only task success, but also whether policies genuinely condition actions on language.

### Acknowledgements

This work is supported in part by the National Natural Science Foundation of China under Grant U23A20300; in part by the Beijing Zhongguancun Academy (Grant No. C20250510); in part by the High Innovation Plan (Grant No. 202504841022).

### Impact Statement

This paper presents work whose goal is to advance the field of Machine Learning. There are many potential societal consequences of our work, none which we feel must be specifically highlighted here.

### References

Bai, S., Cai, Y., Chen, R., Chen, K., Chen, X., Cheng, Z., Deng, L., Ding, W., Gao, C., Ge, C., Ge, W., Guo, Z., Huang, Q., Huang, J., Huang, F., Hui, B., Jiang, S., Li, Z., Li, M., Li, M., Li, K., Lin, Z., Lin, J., Liu, X., Liu, J., Liu, C., Liu, Y., Liu, D., Liu, S., Lu, D., Luo, R., Lv, C., Men, R., Meng, L., Ren, X., Ren, X., Song, S., Sun, Y., Tang, J., Tu, J., Wan, J., Wang, P., Wang, P., Wang, Q., Wang, Y., Xie, T., Xu, Y., Xu, H., Xu, J., Yang, Z., Yang, M., Yang, J., Yang, A., Yu, B., Zhang, F., Zhang, H., Zhang, X., Zheng, B., Zhong, H., Zhou, J., Zhou, F., Zhou, J., Zhu, Y., and Zhu, K. Qwen3-VL technical report. arXiv preprint arXiv:2511.21631, 2025.

Bi, H., Wu, L., Lin, T., Tan, H., Su, Z., Su, H., and Zhu, J. H-RDT: Human manipulation enhanced bimanual robotic manipulation. arXiv preprint arXiv:2507.23523, 2025.

Bjorck, J., Blukis, V., neda, F. C., Cherniadev, N., Da, X., Ding, R., Fan, L. J., Fang, Y., Fox, D., Hu, F., Huang, S., Jang, J., Jiang, X., Kundalia, K., Kautz, J., Li, Z., Lin, K., Lin, Z., Magne, L., Man, Y., Mandlekar, A., Narayan, A., Nasiriany, S., Reed, S., Tan, Y. L., Wang, G., Wang, J., Wang, Q., Wang, S., Xiang, J., Xie, Y., Xu, Y., Ye, S., Yu, Z., Zhao, Y., Zhang, Z., Zheng, R., and Zhu, Y. GR00T N1.5: An improved open foundation model for generalist humanoid robots. https://research.nvidia.

com/labs/gear/gr00t-n1_5/, June 2025a. Accessed: 2026-01-19.

Bjorck, J., Casta˜neda, F., Cherniadev, N., Da, X., Ding, R., Fan, L., Fang, Y., Fox, D., Hu, F., Huang, S., et al. GR00T N1: An open foundation model for generalist humanoid robots. arXiv preprint arXiv:2503.14734, 2025b.

Black, K., Brown, N., Driess, D., Esmail, A., Equi, M., Finn, C., Fusai, N., Groom, L., Hausman, K., Ichter, B., et al. π0: A vision-language-action flow model for general robot control. arXiv preprint arXiv:2410.24164, 2024.

Y., Leal, I., Lee, K.-H., Levine, S., Lu, Y., Malla, U., Manjunath, D., Mordatch, I., Nachum, O., Parada, C., Peralta, J., Perez, E., Pertsch, K., Quiambao, J., Rao, K., Ryoo, M., Salazar, G., Sanketi, P., Sayed, K., Singh, J., Sontakke, S., Stone, A., Tan, C., Tran, H., Vanhoucke, V., Vega, S., Vuong, Q., Xia, F., Xiao, T., Xu, P., Xu, S., Yu, T., and Zitkovich, B. Rt-1: Robotics transformer for real-world control at scale. In arXiv preprint arXiv:2212.06817, 2022.

Bu, Q., Cai, J., Chen, L., Cui, X., Ding, Y., Feng, S., Gao, S., He, X., Huang, X., Jiang, S., et al. Agibot world colosseo: A large-scale manipulation platform for scalable and intelligent embodied systems. arXiv preprint arXiv:2503.06669, 2025.

Cai, J., Cai, Z., Cao, J., Chen, Y., He, Z., Jiang, L., Li, H., Li, H., Li, Y., Liu, Y., Lu, Y., Lv, Q., Ma, H., Pang, J., Qiao, Y., Qiu, Z., Shen, Y., Shi, X., Tian, Y., Wang, B., Wang, H., Wang, J., Wang, T., Wei, X., Wu, C., Xie, Y., Xing, B., Yang, Y., Yang, Y., Yu, Q., Yuan, F., Zeng, J., Zhang, J., Zhang, S., Zhang, S., Zhaxi, Z., Zhou, B., Zhou, Y., Zhou, Y., Zhu, H., Zhu, Y., and Zhu, Y. InternVLA-A1: Unifying understanding, generation and action for robotic manipulation. arXiv preprint arXiv:2601.02456, 2026.

Cai, X., Qiu, R.-Z., Chen, G., Wei, L., Liu, I., Huang, T., Cheng, X., and Wang, X. In-N-On: Scaling egocentric manipulation with in-the-wild and on-task data. arXiv preprint arXiv:2511.15704, 2025.

Chen, T., Chen, Z., Chen, B., Cai, Z., Liu, Y., Li, Z., Liang, Q., Lin, X., Ge, Y., Gu, Z., et al. Robotwin 2.0: A scalable data generator and benchmark with strong domain randomization for robust bimanual robotic manipulation. arXiv preprint arXiv:2506.18088, 2025.

Community, S. Starvla: A lego-like codebase for visionlanguage-action model developing. arXiv preprint arXiv:2604.05014, 2026.

Black, K., Brown, N., Darpinian, J., Dhabalia, K., Driess, D., Esmail, A., Equi, M., Finn, C., Fusai, N., Galliker, M. Y., Ghosh, D., Groom, L., Hausman, K., Ichter, B., Jakubczak, S., Jones, T., Ke, L., LeBlanc, D., Levine, S., Li-Bell, A., Mothukuri, M., Nair, S., Pertsch, K., Ren, A. Z., Shi, L. X., Smith, L., Springenberg, J. T., Stachowicz, K., Tanner, J., Vuong, Q., Walke, H., Walling,

- A., Wang, H., Yu, L., and Zhilinsky, U. π0.5: a visionlanguage-action model with open-world generalization. arXiv preprint arXiv:2504.16054, 2025.

Brohan, A., Brown, N., Carbajal, J., Chebotar, Y., Dabis, J., Finn, C., Gopalakrishnan, K., Hausman, K., Herzog, A., Hsu, J., Ibarz, J., Ichter, B., Irpan, A., Jackson, T., Jesmonth, S., Joshi, N., Julian, R., Kalashnikov, D., Kuang,

Fu, Y., Chen, N., Zhao, J., Shan, S., Yao, G., Wang, P., Wang, Z., and Zhang, S. METIS: Multi-source egocentric training for integrated dexterous vision-language-action model. arXiv preprint arXiv:2511.17366, 2025.

GEAR-Team, Azzolini, A., Bjorck, J., Blukis, V., Casta˜neda, F., Chand, R., et al. Gr00t n1.6: An improved open foundation model for generalist humanoid robots. https://research.nvidia.

com/labs/gear/gr00t-n1_6/, December 2025.

Hancock, A. J., Wu, X., Zha, L., Russakovsky, O., and Majumdar, A. Actions as language: Fine-tuning vlms into vlas without catastrophic forgetting. arXiv preprint arXiv:2509.22195, 2025.

Khazatsky, A., Pertsch, K., Nair, S., Balakrishna, A., Dasari, S., Karamcheti, S., Nasiriany, S., Srirama, M. K., Chen, L. Y., Ellis, K., et al. Droid: A large-scale in-the-wild robot manipulation dataset. arXiv preprint arXiv:2403.12945, 2024.

Kim, M. J., Pertsch, K., Karamcheti, S., Xiao, T., Balakrishna, A., Nair, S., Rafailov, R., Foster, E. P., Sanketi, P. R., Vuong, Q., Kollar, T., Burchfiel, B., Tedrake, R., Sadigh, D., Levine, S., Liang, P., and Finn, C. OpenVLA: An open-source vision-language-action model. In Annual Conference on Robot Learning (CoRL), 2024.

Kim, M. J., Finn, C., and Liang, P. Fine-tuning visionlanguage-action models: Optimizing speed and success. arXiv preprint arXiv:2502.19645, 2025.

Li, Q., Liang, Y., Wang, Z., Luo, L., Chen, X., Liao, M., Wei, F., Deng, Y., Xu, S., Zhang, Y., et al. CogACT: A foundational vision-language-action model for synergizing cognition and action in robotic manipulation. arXiv preprint arXiv:2411.19650, 2024a.

Li, X., Hsu, K., Gu, J., Mees, O., Pertsch, K., Walke, H. R., Fu, C., Lunawat, I., Sieh, I., Kirmani, S., Levine, S., Wu, J., Finn, C., Su, H., Vuong, Q., and Xiao, T. SimplerEnv: Evaluating real-world robot manipulation policies in simulation. In Annual Conference on Robot Learning (CoRL), 2024b.

Li, X., Li, P., Liu, M., Wang, D., Liu, J., Kang, B., Ma, X., Kong, T., Zhang, H., and Liu, H. Towards generalist robot policies: What matters in building vision-language-action models. arXiv preprint arXiv:2412.14058, 2024c.

- Lin, X., Lian, S., Yu, B., Yang, R., Wu, C., Miao, Y., Jin, Y., Shi, Y., Huang, C., Cheng, B., et al. PhysBrain: Human egocentric data as a bridge from vision language models to physical intelligence. arXiv preprint arXiv:2512.16793, 2025.
- Lin, Y., Wang, A. S., Sutanto, G., Rai, A., and Meier, F. Polymetis. https://facebookresearch.github. io/fairo/polymetis/, 2021.

Liu, B., Zhu, Y., Gao, C., Feng, Y., Liu, Q., Zhu, Y., and Stone, P. LIBERO: Benchmarking knowledge transfer for lifelong robot learning. Advances in neural information processing systems (NeurIPS), 36:44776–44791, 2023.

Liu, S., Wu, L., Li, B., Tan, H., Chen, H., Wang, Z., Xu, K., Su, H., and Zhu, J. RDT-1b: a diffusion foundation model for bimanual manipulation. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/forum? id=yAzN4tz7oI.

Liu, X., Gong, C., et al. Flow straight and fast: Learning to generate and transfer data with rectified flow. In International Conference on Learning Representations (ICLR), 2022.

Loshchilov, I. and Hutter, F. Decoupled weight decay regularization. In International Conference on Learning Representations (ICLR), 2017.

Lv, Q., Kong, W., Li, H., Zeng, J., Qiu, Z., Qu, D., Song, H., Chen, Q., Deng, X., and Pang, J. F1: A vision-languageaction model bridging understanding and generation to actions. arXiv preprint arXiv:2509.06951, 2025.

Nasiriany, S., Maddukuri, A., Zhang, L., Parikh, A., Lo, A., Joshi, A., Mandlekar, A., and Zhu, Y. RoboCasa: Large-scale simulation of everyday tasks for generalist robots. In Robotics: Science and Systems, 2024.

O’Neill, A., Rehman, A., Maddukuri, A., Gupta, A., Padalkar, A., Lee, A., Pooley, A., Gupta, A., Mandlekar, A., Jain, A., et al. Open x-embodiment: Robotic learning datasets and rt-x models: Open x-embodiment collaboration. In 2024 IEEE International Conference on Robotics and Automation (ICRA), pp. 6892–6903. IEEE, 2024.

Peebles, W. and Xie, S. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pp. 4195–4205, October 2023.

Pertsch, K., Stachowicz, K., Ichter, B., Driess, D., Nair, S., Vuong, Q., Mees, O., Finn, C., and Levine, S. FAST: Efficient action tokenization for vision-language-action models. arXiv preprint arXiv:2501.09747, 2025.

Qu, D., Song, H., Chen, Q., Yao, Y., Ye, X., Ding, Y., Wang, Z., Gu, J., Zhao, B., Wang, D., et al. Spatialvla: Exploring spatial representations for visual-language-action model. arXiv preprint arXiv:2501.15830, 2025.

Rasley, J., Rajbhandari, S., Ruwase, O., and He, Y. Deepspeed: System optimizations enable training deep learning models with over 100 billion parameters. In Proceedings of the 26th ACM SIGKDD international conference on knowledge discovery & data mining, pp. 3505–3506, 2020.

Shen, Y., Wei, F., Du, Z., Liang, Y., Lu, Y., Yang, J., Zheng, N., and Guo, B. VideoVLA: Video generators can be generalizable robot manipulators. In Advances in neural information processing systems (NeurIPS), 2025. URL https://openreview.net/forum? id=UPHlqbZFZB.

Walke, H. R., Black, K., Zhao, T. Z., Vuong, Q., Zheng, C., Hansen-Estruch, P., He, A. W., Myers, V., Kim, M. J., Du, M., et al. Bridgedata v2: A dataset for robot learning at

scale. In Annual Conference on Robot Learning (CoRL), pp. 1723–1736. PMLR, 2023.

Xing, Y., Luo, X., Xie, J., Gao, L., Shen, H. T., and Song, J. Shortcut learning in generalist robot policies: The role of dataset diversity and fragmentation. In Conference on Robot Learning, pp. 3239–3266. PMLR, 2025.

Xu, K., Zhao, S., Zhou, Z., Li, Z., Pi, H., Zhu, Y., Wang, Y., and Xiong, R. A joint modeling of vision-language-action for target-oriented grasping in clutter. arXiv preprint arXiv:2302.12610, 2023.

Xu, K., Zhu, Z., Chen, A., Zhao, S., Huang, Q., Yang, Y., Lu, H., Xiong, R., Tomizuka, M., and Wang, Y. Seeing to act, prompting to specify: A bayesian factorization of vision language action policy. arXiv preprint arXiv:2512.11218, 2025.

Yang, J., Tan, R., Wu, Q., Zheng, R., Peng, B., Liang, Y., Gu, Y., Cai, M., Ye, S., Jang, J., et al. Magma: A foundation model for multimodal ai agents. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 14203–14214, 2025a.

Yang, Y., Li, X., Chen, Y., Song, J., Wang, Y., Xiao, Z., Su, J., Qiaoben, Y., Liu, P., and Deng, Z. Mantis: A versatile vision-language-action model with disentangled visual foresight. arXiv preprint arXiv:2511.16175, 2025b.

Yu, B., Lian, S., Lin, X., Shen, Z., Wei, Y., Liu, H., Wu, C., Yuan, H., Wang, B., Huang, C., et al. 3d-mix for vla: A plug-and-play module for integrating vggt-based 3d information into vision-language-action models. arXiv preprint arXiv:2603.24393, 2026a.

Yu, B., Lian, S., Lin, X., Wei, Y., Shen, Z., Wu, C., Miao, Y., Wang, X., Wang, B., Huang, C., and Chen, K. TwinBrainVLA: Unleashing the potential of generalist vlms for embodied tasks via asymmetric mixture-of-transformers. arXiv preprint arXiv:2601.14133, 2026b.

Zheng, J., Li, J., Wang, Z., Liu, D., Kang, X., Feng, Y., Zheng, Y., Zou, J., Chen, Y., Zeng, J., et al. X-VLA: Soft-prompted transformer as scalable crossembodiment vision-language-action model. arXiv preprint arXiv:2510.10274, 2025a.

Zheng, R., Liang, Y., Huang, S., Gao, J., III, H. D., Kolobov,

- A., Huang, F., and Yang, J. TraceVLA: Visual trace prompting enhances spatial-temporal awareness for generalist robotic policies. arXiv preprint arXiv:2412.10345, 2025b.

Zhou, Z., Zhu, Y., Wen, J., Shen, C., and Xu, Y. Visionlanguage-action model with open-world embodied reasoning from pretrained knowledge. arXiv preprint arXiv:2505.21906, 2025.

### A. Derivation of the LLR Objective

In this section, we provide the derivation for the Log-Likelihood Ratio (LLR) objective used in LangForce. Our core motivation is to maximize the learning signal discussed previously: the Conditional Pointwise Mutual Information (PMI) between the action a and the language instruction ℓ, given the visual observation v. The PMI is formally defined as:

π(a,ℓ | v) p(a | v)p(ℓ | v)

PMI(a,ℓ | v) = log

. (10)

By applying the chain rule of probability π(a,ℓ | v) = π(a | v,ℓ)p(ℓ | v), we can rewrite the PMI as the log-ratio between the posterior policy and the vision-only prior:

π(a | v,ℓ)p(ℓ | v) p(a | v)p(ℓ | v)

π(a | v,ℓ) p(a | v)

PMI(a,ℓ | v) = log

. (11)

= log

This formulation highlights that maximizing PMI is equivalent to maximizing the divergence between the languageconditioned policy and the vision-only prior, directly penalizing the collapse where π(a | v,ℓ) ≈ p(a | v).

Alternatively, using the chain rule π(a,ℓ | v) = p(ℓ | a,v)p(a | v), we arrive at the second form, which constitutes our practical LLR objective:

p(ℓ | a,v)p(a | v) p(a | v)p(ℓ | v)

p(ℓ | a,v) p(ℓ | v)

PMI(a,ℓ | v) = log

= log p(ℓ | a,v) − log p(ℓ | v). (12)

= log

This objective represents the difference between the log-likelihood of the instruction given the action and vision, and the log-likelihood of the instruction given vision alone. Maximizing this quantity compels the model to select actions a that make the instruction ℓ significantly more probable than determining it from the visual context v alone, thereby extracting the additional information required to solve ambiguous or out-of-distribution tasks.

### B. Additional Experiments

#### B.1. Additional Experiments on LIBERO

- Table 6. Comparison on the LIBERO benchmark. We train one policy for all 4 suites. Avg@500 success rates (%) across four task suites: Spatial, Object, Goal, and Long.

Method Spatial Object Goal Long Avg OpenVLA (Kim et al., 2024) 87.4 88.4 79.2 53.7 76.5 OpenVLA-OFT (Kim et al., 2025) 97.6 98.4 97.9 94.5 97.1 π0 (Black et al., 2024) 96.8 98.8 95.8 85.2 94.1 π0.5 (Black et al., 2025) 98.8 98.2 98.0 92.4 96.9 Qwen3-VL-FAST 97.3 97.4 96.3 90.6 95.4 Qwen3-VL-OFT 97.8 98.6 96.2 93.8 96.6 Qwen3-VL-GR00T 97.8 98.8 97.4 92.0 96.5 VisionOnly Qwen3-VL-GR00T 90.2 99.6 9.8 86.0 71.4 LangForce 99.2 99.6 99.4 95.2 98.4

We also evaluate LangForce on the LIBERO benchmark (Liu et al., 2023). Given that the training and testing environments in LIBERO are highly similar and current VLA research has largely saturated this benchmark (with baselines exceeding 95%), our method yields comparable performance on the Spatial, Object, and Long suites. However, on the Goal suite, LangForce achieves a success rate of 99.4%, outperforming the Qwen3-VL-GR00T baseline (97.4%) by +2.0%. As highlighted in Section 2, the Goal suite features significant visual ambiguity where multiple tasks share the same scene. This result empirically validates that our method effectively mitigates the vision shortcut, enabling the model to resolve ambiguity through robust instruction following.

To further substantiate this claim, we conducted a quantitative analysis of the conditional entropy H(ℓ | v) on the LIBERO Goal dataset. We approximated this metric by computing the Negative Log-Likelihood (NLL) of the ground-truth instructions given the visual observations across 40,000 samples. The results are summarized in Table 7.

- Table 7. Quantitative analysis of conditional entropy on LIBERO Goal. We report the Negative Log-Likelihood (NLL) and Perplexity (PPL) of instructions given visual observations, serving as a proxy for H(ℓ | v). Higher NLL/PPL indicates that the model preserves the necessary uncertainty about the task given only vision, preventing the information collapse observed in baselines.

Method NLL (nats/token) ↑ PPL (exp(NLL)) ↑ Std Dev (NLL)

QwenGR00T (Baseline) 8.51 4964.1 0.55 LangForce 9.47 12964.9 0.54

LangForce achieves a significantly higher NLL (9.47 nats/token) compared to the QwenGR00T baseline (8.51 nats/token). This difference is even more pronounced in terms of Perplexity (PPL), where LangForce reaches 12964.9 compared to the baseline’s 4964.1. It is worth noting that the language structure in LIBERO Goal is highly repetitive (e.g., “put the [object] in/on the [receptacle]”), which naturally encourages the model to fit these syntactic patterns. Despite this, our method achieves a higher NLL and PPL, indicating that the uncertainty stems primarily from the task-specific nouns (objects and receptacles) rather than the sentence structure. In the context of LIBERO Goal, where visual scenes are inherently ambiguous, the baseline’s lower NLL implies it is “overconfident” in predicting these key nouns solely from vision. This confirms that standard training leads to a collapse in conditional entropy, where the model learns spurious correlations (v → ℓ) and ignores the actual need for language disambiguation. In contrast, the higher NLL observed in LangForce indicates that our model prevents the pathological collapse of H(ℓ | v) observed in standard training, preserving uncertainty levels that align with the inherent ambiguity of visual scenes. By maintaining this necessary uncertainty, our method forces the policy to actively utilize the provided language instruction to resolve ambiguity, directly contributing to the superior performance on the Goal suite.

B.2. Additional Experiments on RoboCasa

More quantitative results on RoboCasa are presented in Table 9. Consistent with the empirical evidence presented in our motivation (Section 2), the VisionOnly baseline achieves a surprisingly high success rate of 44.7%, lagging only slightly behind the standard QwenGR00T baseline (47.8%). This observation reconfirms the prevalence of the vision shortcut in this benchmark, suggesting that a significant portion of tasks can be solved by relying solely on visual cues. However, LangForce breaks this performance ceiling, achieving a state-of-the-art average success rate of 52.6% and surpassing all competing baselines, including QwenOFT (48.8%), Isaac-GR00T N1.5 (48.2%), and Isaac-GR00T N1.6 (47.6%). Crucially, our method demonstrates substantial gains in tasks where the vision-only policy falters. For instance, in “PnP Novel From Placemat To Bowl”, LangForce reaches 62.0%, substantially surpassing both the VisionOnly baseline (32.0%) and QwenGR00T (44.0%). These results indicate that maximizing the LLR objective successfully forces the policy to extract and utilize task-specifying information from language, rather than settling for local optima based on visual shortcuts.

- Table 8. Selected OXE Subsets for Pre-training. We curate 20 subsets from the Open X-Embodiment dataset that utilize end-effector position control. The table details the dataset source, the robot platform, and the number of episodes utilized.

Dataset Name Robot Episodes Dataset Name Robot Episodes

RT-1 Robot Action Google Robot 79,499 BC-Z Google Robot 39,350 Berkeley Bridge WidowX 25,460 Berkeley Fanuc Manipulation Fanuc Mate 415 Berkeley Autolab UR5 UR5 896 UCSD Kitchen xArm 150 USC Jaco Play Jaco 2 976 Stanford Kuka Multimodal Kuka iiwa 3,000 CMU Stretch Hello Stretch 135 DobbE Hello Stretch 5,208 NYU VINN Hello Stretch 435 - - -

DROID Franka 92,233 Austin BUDS Franka 50 Freiburg Franka Play Franka 3,242 CMU Franka Pick-Insert Data Franka 520 Stanford HYDRA Franka 550 Austin Mutex Franka 1,500 Austin VIOLA Franka 135 CMU Play Fusion Franka 576 NYU Franka Play Franka 456 - - -

Total 254,786

#### B.3. Real-World Experiments Setup

We evaluate LangForce on a real-world robotic setup using a Franka Research 3 arm equipped with an Intel RealSense D435 camera, as visualized in Figure 8. To assess the model’s instruction-following capabilities and generalization, we design a “Pick and Place All Blocks” task where the robot must pick up all specific colored blocks (Red, Blue, Yellow, or

- Table 9. Results of evaluating the VLA models with the GR1 robot in the RoboCasa Tabletop simulation environment. The results for Isaac-GR00T N1.5 and Isaac-GR00T N1.6 are sourced from the official Isaac-GR00T github repository (Bjorck et al., 2025b). The results for QwenGR00T, QwenPI, QwenOFT, QwenFAST are sourced from the official starVLA experiments (Community, 2026). We highlight the best results in bold and the second-best results with underline.

Task Isaac-GR00TN1.5 Isaac-GR00TN1.6 QwenGR00T+Qwen3VL +Qwen3VLQwenPI +Qwen3VLQwenOFT +Qwen3VLQwenFAST QwenGR00TVisionOnly +Qwen3VLLangForce

PnP Bottle To Cabinet Close 54.0 51.5 46.0 26.0 30.0 38.0 70.0 72.0 PnP Can To Drawer Close 50.0 13.0 80.0 62.0 76.0 44.0 78.0 78.0 PnP Cup To Drawer Close 38.0 8.5 54.0 42.0 44.0 56.0 42.0 46.0 PnP Milk To Microwave Close 60.0 14.0 48.0 50.0 44.0 44.0 50.0 56.0 PnP Potato To Microwave Close 32.0 41.5 28.0 42.0 32.0 14.0 44.0 36.0 PnP Wine To Cabinet Close 38.0 16.5 46.0 32.0 36.0 14.0 40.0 46.0

PnP * to * Close (Avg) 45.3 24.2 50.3 42.3 43.7 35.0 54.0 55.7 PnP Novel From Cuttingboard To Basket 38.0 58.0 48.0 40.0 50.0 54.0 58.0 66.0 PnP Novel From Cuttingboard To Cardboardbox 46.0 46.5 40.0 46.0 40.0 42.0 26.0 40.0 PnP Novel From Cuttingboard To Pan 58.0 68.5 68.0 60.0 70.0 58.0 72.0 68.0 PnP Novel From Cuttingboard To Pot 62.0 65.0 52.0 40.0 54.0 58.0 50.0 48.0 PnP Novel From Cuttingboard To Tieredbasket 28.0 46.5 56.0 44.0 38.0 40.0 20.0 44.0 PnP Novel From Cuttingboard To * (Avg) 46.4 56.9 52.8 46.0 50.4 50.4 45.2 53.2 PnP Novel From Placemat To Basket 30.0 58.5 42.0 44.0 32.0 36.0 48.0 54.0 PnP Novel From Placemat To Bowl 60.0 57.5 44.0 52.0 58.0 38.0 32.0 62.0 PnP Novel From Placemat To Plate 56.0 63.0 48.0 50.0 52.0 42.0 34.0 52.0 PnP Novel From Placemat To Tieredshelf 36.0 28.5 18.0 28.0 24.0 18.0 16.0 24.0 PnP Novel From Placemat To * (Avg) 45.5 51.9 38.0 43.5 41.5 33.5 32.5 48.0 PnP Novel From Tray To Cardboardbox 52.0 51.5 38.0 34.0 44.0 28.0 50.0 50.0 PnP Novel From Tray To Plate 48.0 71.0 56.0 64.0 56.0 34.0 64.0 58.0 PnP Novel From Tray To Pot 60.0 64.5 50.0 44.0 62.0 46.0 52.0 62.0 PnP Novel From Tray To Tieredbasket 52.0 57.0 36.0 50.0 54.0 36.0 42.0 44.0 PnP Novel From Tray To Tieredshelf 32.0 31.5 16.0 28.0 30.0 16.0 16.0 22.0 PnP Novel From Tray To * (Avg) 48.8 55.1 39.2 44.0 49.2 32.0 44.8 47.2 PnP Novel From Plate To Bowl 58.0 57.0 60.0 52.0 60.0 52.0 26.0 54.0 PnP Novel From Plate To Cardboardbox 44.0 43.5 50.0 40.0 50.0 30.0 38.0 48.0 PnP Novel From Plate To Pan 60.0 51.0 54.0 36.0 66.0 48.0 44.0 54.0 PnP Novel From Plate To Plate 64.0 78.7 70.0 48.0 68.0 50.0 60.0 78.0 PnP Novel From Plate To * (Avg) 56.5 57.6 58.5 44.0 61.0 45.0 42.0 58.5 Average 48.2 47.6 47.8 43.9 48.8 39.0 44.7 52.6

Green) in the scene and place them into a box. For the training set, we collect 100 expert demonstrations for each of the Green, Blue, and Yellow blocks. The Red block is held out to test Out-Of-Distribution (OOD) generalization. Crucially, during data collection, we ensure that multiple colored blocks are present in the scene simultaneously. During testing, we further introduce distractors such as cabinets and wrenches into the environment to rigorously evaluate robustness. This setup forces the model to rely on language instructions to identify the correct target, rather than exploiting simple visual shortcuts (e.g., reacting to the only object in the scene). We initialize our model using a subset of the Open X-Embodiment (OXE) dataset, comprising 20 subsets with end-effector (EEF) control configurations. Detailed statistics of the training datasets are provided in Table 8. To align with the action space of the pre-training data, we collect data and control the robot using end-effector (EEF) pose control. We also provide four qualitative inference examples for the real-world vegetable pick-and-place task in Figure 9.

#### B.4. Preservation of General Capabilities

It is important to make a nuanced distinction regarding these results: while LangForce effectively preserves the model’s normal text-only conversational abilities, its general vision-language conversation skills (involving both image and text inputs) may still experience some degradation following VLA training. We hypothesize that this is primarily because our training setup requires the vision tower and multimodal projection layers to adapt for control tasks, inevitably shifting visual representations away from the original pre-trained vision-language alignment manifold. Such specialization is a natural consequence of optimizing an embodied agent for action execution.

[Figure 31]

[Figure 32]

What is LLM? Tell me briefly.

###### LLM in LangForce

LLMs are large language models — essentially, they’re AI systems trained on vast amounts of internet

text to understand, generate, and translate human languages effectively.They can answer questions, write stories, code, emails, etc., often producing surprisingly fluent and context-aware outputs.Think of them as supercharged versions of Google Translate or autocomplete tools — but much smarter and more flexible.

[Figure 33]

[Figure 34]

###### Qwen3-VL-4B-Instruct

[Figure 35]

Example:

LLM stands for Large Language Model. It’s a type of artificial intelligence model trained on vast amounts of text data to understand and generate humanlike language. LLMs can answer questions, write stories, code, translate languages, and more — often mimicking human conversation and style. Examples include GPT-3, GPT-4, and Claude.

> Input: “Write a short story about a robot learning to bake cookies.” > Output: A full-length imaginative narrative written in natural language. They’re also used in education, customer service, gaming, legal/political analysis, etc.

TL;DR: LLMs = giant AI brains fueled by internet memes, articles, books — all tuned to mimic human

creativity and reasoning. (Note: Some are copyrighted or require licenses depending on use case.)

[Figure 36]

A large amount of meaningless, infinitely repeated Chinese text.

###### LLM in QwenGR00T

Large-scale machine learning models.\n\n简而言之，LLMs are large-scale machine learning models — often referring to transformer-based models like GPT或多层堆叠的硅基结构。\n\n

但他们’re not just堆叠硅基结构.‘\n\n要害之处在于他们的训练数据量级和参数量级都达到了前所未有的规模，\n\n并且他们的训练过程本身就构成了某种意义上的文明进步图谱。

\n\n她们不仅仅是工具箱里的工具，\n\n她们是文明本身。\n\n理解这句话的最后一句话？\n\n这句话的最后一句话是：\n\n她们不仅仅是工具箱里的工具…

Figure 5. Additional qualitative comparison. Demonstrating the preservation of general VLM capabilities on another example.

Crucially, however, LangForce prevents the collapse of the backbone’s text-only language behavior. In standard baselines, the prevalence of vision shortcuts often renders instruction tokens effectively redundant for control. This weakens the training signal that reinforces meaningful language processing, leading to a drift in shared parameters that manifests as failures even on pure text queries (as seen in Fig. 4 and Fig. 5). Conversely, our method’s LLR objective enforces a strong, explicit dependency on language. This acts as a regularizer, maintaining the functional utility of instruction tokens and thereby preserving the backbone’s text-only conversational ability, even as the visual modality specializes for control. This preservation is of significant practical value, ensuring that the VLM backbone does not degenerate into a shallow feature mapper. By retaining its linguistic core, the agent preserves the potential for higher-level reasoning and generalization to novel instructions—key motivations for employing Foundation Models in robotics.

- Table 10. Ablation on auxiliary language prediction. Adding an auxiliary p(ℓ | v) objective to QwenGR00T provides only limited policy benefit and remains far below LangForce, indicating that preserving generic language behavior alone is insufficient unless language is tied to action-relevant distinctions.

Method Spoon on Towel Carrot on Plate Stack Block Eggplant in Basket Avg QwenGR00T 87.5 50.0 29.2 54.2 55.2 + Aux. p(ℓ | v) 79.6 61.6 16.6 69.7 56.9 LangForce 89.6 63.8 33.3 79.2 66.5

Auxiliary language-prediction ablation. To separate generic language retention from action-relevant language grounding, we additionally evaluate a control variant that keeps the same QwenGR00T backbone but adds an auxiliary languageprediction objective for p(ℓ | v) during VLA fine-tuning. This is close to adding an extra SFT-style loss that encourages the model to reconstruct the training instruction from the visual scene. As shown in Table 10, this variant provides only a small policy gain over QwenGR00T (56.9% vs. 55.2%) and remains substantially below LangForce (66.5%). Qualitatively, as shown in Fig. 6, the auxiliary-loss model can preserve surface fluency on text-only prompts, but it often overfits to robot-training instructions, producing command-like or scene-specific responses to open-domain questions. This behavior is expected: optimizing p(ℓ | v) strengthens scene-to-instruction prediction and can sharpen the vision shortcut. In contrast, the LLR objective preserves language by keeping instruction tokens causally useful for action generation, binding language to action-relevant distinctions rather than merely asking the model to reproduce likely training instructions.

|[Figure 37]<br><br>What is LLM?| |
|---|---|
|LLM in QwenGR00T with Auxiliary Loss 𝒑(𝓵 | 𝒗)<br><br>[Figure 38]| |
|Pick up the broccoli and place it on the right of the blue brush| |

###### LLM in LangForce

LLM stands for Large Language Models, which are artificial intelligence systems trained extensively onvast amounts of textual data to understand, interpret, and generate human-like text. Key Characteristics of LLMs:

- • Generative Capabilities: They can produce coherent, contextually relevant written content such as articles, stories, emails, code, etc.
- • Fine-Tuned Performance: Often pretrained on enormous datasets (like Wikipedia, books, internet text) and then fine-tuned for specific tasks or domains.
- • Zero-shot or Few-shot Reasoning: Many LLMs can answer questions or perform reasoning taskswithout explicit training on those particular inputs.
- • Conversations & Dialogue: They excel ... (Subsequent responses are omitted here)

[Figure 39]

|[Figure 40]<br><br>What is VLA?| |
|---|---|
|LLM in QwenGR00T<br><br>with Auxiliary Loss 𝒑(𝓵 | 𝒗)<br><br>[Figure 41]| |
|VLA is a blue cloth| |

LLM in LangForce

The acronym VLA can refer to several different things depending on the context. Here are some common meanings:

- • Very Large Array (Radio Telescope)

- -Full Name: Very Large Array
- -Location: Socoma, Spain; plus individual dishes elsewhere in the world.
- -Purpose: One of the most powerful radio telescopes in the world, used for deep-space observation, galaxy mapping, pulsars, etc.
- -Note: Often confused with the VLBA (Very Long Baseline Array), which is a different systemcombining telescopes across the globe.

- • Virtual Library Architecture ... (Subsequent responses are omitted here)

[Figure 42]

- Figure 6. Qualitative examples for the auxiliary language-prediction ablation. Adding an auxiliary p(ℓ | v) loss helps preserve surface conversational fluency, but the model tends to overfit toward robot-training instructions or scene-specific associations.

#### B.5. Ablation on λ and β in Loss Function

We analyze the impact of hyperparameters λ and β based on the results in Table 11 and Table 12. First, regarding the prior loss weight λ, we observe that even when λ = 0 (where the prior branch is used solely for LLR calculation without explicit action supervision), the model achieves an average success rate of 63.3%. This represents a substantial improvement over the QwenGR00T baseline (55.2%), confirming that the LLR objective alone effectively regularizes the policy. Increasing λ to 0.3 yields the optimal performance of 66.5%, validating that explicitly learning the vision-only prior p(a|v) further aids the decomposition. The performance remains robust across λ ∈ [0,0.5], though it drops slightly at higher values.

Second, for the LLR weight β, setting β = 0 corresponds to training the dual-branch architecture without the mutual information maximization term. This configuration achieves 61.3%, surpassing the baseline (55.2%). This suggests that the architectural design, which explicitly separates a vision-only pathway to absorb dataset biases, inherently helps the posterior branch focus on language. Incorporating the LLR term (β = 0.1) further boosts the success rate to 66.5%.

- Table 11. Ablation study on β in Eq. 9. All experiments are based on the Qwen3-VL-4B backbone.

λ Put Spoon on Towel Put Carrot on Plate Stack Block Eggplant in Basket Avg 0 86.4 64.6 26.0 76.0 63.3 0.1 84.2 65.6 27.3 81.9 64.8 0.3 89.6 63.8 33.3 79.2 66.5 0.5 81.2 61.5 25.8 87.7 64.1

- Table 12. Ablation study on β in Eq. 9. All experiments are based on the Qwen3-VL-4B backbone.

β Put Spoon on Towel Put Carrot on Plate Stack Block Eggplant in Basket Avg 0 80.8 61.0 30.8 72.5 61.3

- 0.1 89.6 63.7 33.3 79.2 66.5
- 0.2 82.5 67.5 34.2 77.9 65.5
- 0.3 73.8 58.3 29.2 85.2 61.6

#### B.6. Ablation on Number of Latent Action Queries

To explore the potential of Latent Action Queries as a out-of-the-box technique, we conduct this ablation solely on the query mechanism, without incorporating our proposed Bayesian decomposition. We analyse the impact of the number of latent action queries on model performance, as summarized in Table 13. We observe that increasing the number of queries from 16 to 64 leads to a substantial improvement in success rate. However, doubling the queries to 128 results in performance saturation without providing additional benefits. Consequently, we adopt 64 queries as the optimal configuration to balance performance and computational overhead.

Table 13. Ablation study on Number of Latent Action Query. All experiments are based on the Qwen3-VL-4B backbone.

Number Put Spoon on Towel Put Carrot on Plate Stack Block Eggplant in Basket Avg 16 69.8 47.9 20.8 60.4 49.7 32 62.5 57.4 25.8 79.2 56.2 64 74.6 58.3 29.2 67.9 57.5 128 72.9 57.3 29.1 70.8 57.5

### C. Discussion

Based on our analysis of the vision shortcut and the Bayesian decomposition framework, we discuss several potential insights that may guide future research and community practices.

Rethinking Data Collection Strategies. Our experiments suggest that the deterministic mapping from visual scenes to language instructions (H(ℓ | v) ≈ 0) in goal-driven datasets is a significant factor contributing to the vision shortcut. To mitigate this, we hypothesize that a shift in data collection strategies could be beneficial. Prioritizing data collection in ambiguous scenarios—where the task cannot be inferred solely from the initial observation—might naturally increase the conditional entropy of language. By enriching datasets with scenes that support multiple valid tasks, models may be forced to rely more heavily on instructions for disambiguation.

Leveraging Human Data for Robustness. Recently, there has been growing interest in training robot models on large-scale human video data, such as HRDT (Bi et al., 2025), In-N-On (Cai et al., 2025), METIS (Fu et al., 2025), and PhysBrain (Lin et al., 2025). Unlike curated robot datasets, human activities are inherently multimodal and context-dependent; the same environment often hosts a wide variety of behaviors, potentially leading to a less sharp p(ℓ | v). We conjecture that injecting action knowledge from such rich human distributions might help mitigate the information collapse observed in robot-only datasets.

World Models as an Alternative Bayesian Formulation. Beyond the VLM framework focused on in this work, recent studies have also explored adapting World Models for VLA control, as seen in F1-VLA (Lv et al., 2025), Mantis (Yang et al., 2025b), and InternVLA-A1 (Cai et al., 2026). From a theoretical perspective, these approaches can be viewed as an alternative instantiation of the Bayesian rule, specifically performing inverse dynamics on imagined futures. If we consider v as a sequence of past frames v≤t, and treat the future state vt+1 as a latent variable generated by the model (conditioned on ℓ), the action inference can be expressed as:

p(vt+1 | v≤t,a,ℓ)p(a | v≤t,ℓ) p(vt+1 | v≤t,ℓ)

. (13)

p(a | v≤t,vt+1,ℓ) =

Here, the numerator p(vt+1 | v≤t,a,ℓ) represents a world model (forward dynamics) predicting the future state. The term p(a | v≤t,ℓ) serves as an action prior, and the denominator p(vt+1 | v≤t,ℓ) represents the future prediction marginalized over actions. In this formulation, the policy execution involves first “imagining” a desired future vt+1 consistent with ℓ, and then inferring the optimal action a via the equation above. Since world models are typically trained on vast amounts of video data, the predictive distribution (the numerator) is often rich and highly sensitive to the action a. We hypothesize that this sensitivity prevents the collapse of the numerator to the denominator. This suggests that world model-based architectures could offer another robust technical path toward solving the vision shortcut, which we plan to explore in future work.

Train (In BridgeDataV2 and Fractal Dataset)

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

Test (In SimplerEnv Benchmark)

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

Put the eggplant into the sink

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

Put the spoon on the tablecloth

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

Stack the green cube on the yellow cube

- Figure 7. Visualization of the domain gap. Top: Training frames from BridgeDataV2 and Fractal. Bottom: LangForce rollouts in SimplerEnv. More inference videos can be found in the supplementary material.

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

Pick up all green blocks and put them into the gray box.

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

- Figure 8. Diagram of human expert data acquired through teleoperation. We use Polymetis (Lin et al., 2021) for teleoperation and collect human expert data to support imitation learning for VLA models.

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

Pick up the carrot and place it in the brown basket

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

Pick up the carrot and place it in the brown basket

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

Pick up the chili pepper and place it in the brown basket

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

Pick up the eggplant and place it in the brown basket eggplant

- Figure 9. Real-world inference examples for vegetable pick-and-place. We show two rollout sequences where LangForce follows language instructions to pick the specified vegetable and place it into the brown basket.

