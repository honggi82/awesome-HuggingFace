arXiv:2505.03981v1[cs.AI]6May2025

# X-REASONER: Towards Generalizable Reasoning Across Modalities and Domains

Qianchu Liu∗, Sheng Zhang∗, Guanghui Qin∗, Timothy Ossowski, Yu Gu, Ying Jin Sid Kiblawi, Sam Preston, Mu Wei, Paul Vozila, Tristan Naumann, Hoifung Poon Microsoft Research github.com/microsoft/x-reasoner

## Abstract

Recent proprietary models (e.g., o3) have begun to demonstrate strong multimodal reasoning capabilities. Yet, most existing open-source research concentrates on training text-only reasoning models, with evaluations limited to mainly mathematical and general-domain tasks. Therefore, it remains unclear how to effectively extend reasoning capabilities beyond text input and general domains. This paper explores a fundamental research question: Is reasoning generalizable across modalities and domains? Our findings support an affirmative answer: General-domain text-based post-training can enable such strong generalizable reasoning. Leveraging this finding, we introduce X-REASONER, a vision-language model posttrained solely on general-domain text for generalizable reasoning, using a twostage approach: an initial supervised fine-tuning phase with distilled long chainof-thoughts, followed by reinforcement learning with verifiable rewards. Experiments show that X-REASONER successfully transfers reasoning capabilities to both multimodal and out-of-domain settings, outperforming existing state-of-theart models trained with in-domain and multimodal data across various general and medical benchmarks (Figure 1). Additionally, we find that X-REASONER’s performance in specialized domains can be further enhanced through continued training on domain-specific text-only data. Building upon this, we introduce XREASONER-MED, a medical-specialized variant that achieves new state of the art on numerous text-only and multimodal medical benchmarks.

100

Baseline (Qwen-2.5-VL-7B-Instruct)

Previous 7B-scale SOTA

X-Reasoner

X-Reasoner-Med

| |
|---|

| |
|---|

| |
|---|

88.2

90

87.2

X-Reasoner Generalizes Across Domains and Modalities

84.9

80

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

71.6 72.4

Accuracy(%)

70

64.0

62.3

60.0

58.7

60

55.0 56.4 57.3

56.7

56.4

56.3

53.2

52.6

52.1

51.3

50

47.8

47.6

43.0

40.7

38.7

40

30

26.4

23.8

23.8

22.2

20

GSM8K MMLU-Pro MMMU-Pro MMMU (Val) MedQA (4-ops) MMLU-Pro-H MMMU-H MedXpertQA-MM

Text Benchmarks Multimodal Benchmarks (X-Modality)

Medical Text Benchmarks (X-Domain)

Medical MultiModal Benchmarks (X-Domain & X-Modality)

- Figure 1: X-REASONER (■ blue bars), fine-tuned solely on general domain text , shows strong generalization across both modalities (e.g., multimodality ) and domains (e.g., medicine ), surpassing prior SOTA (Table 8) trained with in-domain multimodal data. X-REASONER-MED (■ red bars), its medical-specialized variant, sets new SOTA on numerous medical benchmarks.

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

∗ Equal contributions.

Preprint. Under review.

## 1 Introduction

Reasoning has emerged as a foundational capability in language models, paving the way for a new paradigm known as test-time scaling (OpenAI, 2024). This paradigm allows models to achieve consistent gains in complex reasoning tasks through scaling compute resources during inference. Recent proprietary models, such as o3 (OpenAI, 2025), have begun to demonstrate strong multimodal reasoning capabilities. However, existing open-source research primarily focuses on advancing text-only reasoning, employing post-training techniques such as long CoT (Chain-of-Thought) distillation and reinforcement learning with verifiable rewards (DeepSeek-AI et al., 2025; Hu et al., 2025). While these efforts have resulted in powerful reasoning models, their evaluation has largely been confined to math and general-domain text benchmarks. Consequently, it remains uncertain how reasoning capabilities can effectively generalize beyond text inputs and general-domain contexts.

Recently, a growing body of open-source work has started addressing multimodal reasoning, i.e., extending reasoning to inputs that include both vision and language (Wang et al., 2025; Meng et al., 2025; Guo et al., 2024). Yet, current multimodal approaches often rely heavily on curating multimodal datasets which are tailored to specific tasks or domains, limiting their generalisability.

In this work, we ask a fundamental, yet under-explored question: Is reasoning generalizable across modalities and domains? Specifically, we investigate whether such generalizable reasoning can be achieved through general-domain text-based reasoning post-training. Beyond scientific significance, this question is also motivated by the practical advantages of general domain text-only training: i.e. its compute efficiency and the abundance and verifiability of general-domain textual reasoning data, which together allow us to avoid the cost and complexity of curating multimodal or domain-specific data. Our hypothesis is that text-based post-training, when carefully designed, can impart universal reasoning patterns that robustly transfer across both unseen domains and input modalities.

To investigate our hypothesis, we conduct an extensive empirical study using a two-stage text-only post-training recipe: supervised fine-tuning (SFT) on general-domain text data with distilled long CoTs, followed by reinforcement learning with verifiable rewards (RLVR) using mathematical textual questions. Remarkably, this pure textual training regimen proves sufficient to instill strong reasoning capabilities, enabling high performance not only on general-domain tasks but also on complex multimodal and domain-specific tasks. This finding suggests that the core structure of reasoning can indeed be acquired from general-domain text alone.

Building upon these insights, we introduce X-REASONER, a 7B vision-language model post-trained with the proposed recipe. Despite being trained only for text-based reasoning, X-REASONER achieves superior performance on a suite of both text-only and multimodal reasoning benchmarks, outperforming prior state-of-the-art 7B models that were explicitly trained with multimodal reasoning supervision, on challenging tasks including MMMU, MMMU-Pro and MathVista. We further demonstrate that the improvement from X-REASONER can generalize to specialized domains such as medicine. To boost in-domain performance even further, we explore the impact of incorporating domain-specific textual data. To this end, we introduce X-REASONER-MED, a medical-specialized variant of X-REASONER, which undergoes additional post-training on medical domain text. XREASONER-MED sets new records on numerous textual and multimodal medical tasks.

Our key contributions can be summarized as the following:

#### 1. Generalizable Reasoning Study:

- • We conduct an in-depth study of whether reasoning capabilities trained on general-domain text can generalize across input modalities and domains.
- • We find that combining SFT and RL yields the strongest gains, and that math-based data serves as a strong anchor, though domain-specific data remains important for specialization.

#### 2. Introduction of X-REASONER:

- • We propose an effective post-training recipe for vision-language reasoning models that relies entirely on general-domain text-based data.
- • X-REASONER improves performance across modalities and domains, remarkably, achieving SOTA on challenging general-domain multimodal benchmarks.

#### 3. Medical Domain Extension via X-REASONER-MED:

• We present X-REASONER-MED, a domain-adapted variant of X-REASONER trained on medical text, setting new 7B-scale records across medical text-only and multimodal tasks.

## 2 A Journey Towards Generalizable Reasoning

MMMU Pro Text-only Text-only

Step 1: General-domain Text-only Long-CoT SFT

###### Instruction-tuned Vision-Language Model

Step2: Mathematical Text-only RLVR

43.0

###### X-Reasoner

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

Step 2

Enable reasoning patterns (e.g., self-reflection, selfverification, self-correction)

Capable of following instructions

Improve generalizable reasoning capabilities

40.4

Improved reasoning performance across modalities & domains

Smaller improvements from CoT prompting

Mitigate endless thinking issues from Step 1

Step 1

38.3

Learn general-domain knowledge

Base

- Figure 2: (Left) Our recipe for generalizable reasoning: based on an instruction-tuned VLM, we first conduct SFT on general-domain text data with distilled long CoTs. This is followed by RLVR on mathematical textual questions. This resulting model, X-REASONER, exhibits significantly enhanced reasoning capabilities across modalities and domains. (Right) Model performance on MMMU-Pro (multimodal task) steadily improves at each stage of our recipe.

In this section, we embark on a systematic exploration to understand whether generalizable reasoning can be effectively acquired through general-domain text-based post-training. Here, we define generalizable reasoning as the ability to transfer reasoning capabilities not only across different task distributions but also across diverse modalities and specialized domains. To thoroughly address this question, we break it down into two sub-questions corresponding to two predominant post-training strategies used to foster reasoning: SFT and RL. Starting from an instruction-tuned vision-language model (VLM) capable of following instructions yet benefiting less from CoT prompting (see Section 2.1), we perform a detailed empirical study utilizing these two post-training methods. Our objective is to identify a robust and effective recipe that achieves strong, generalizable reasoning capabilities purely through general-domain text-based training.

#### 2.1 RQ1: Will Text-only SFT Improve Generalizable Reasoning?

Our first research question investigates whether reasoning trained solely via SFT on general-domain textual data can generalize beyond the training distribution. Specifically, we study two key aspects: (1) the extent to which generalization can occur across domains and modalities, and (2) the essential role of reasoning in enabling such generalization.

Extent of Generalization We begin by examining whether reasoning capabilities acquired through text-only SFT can generalize across distributional shifts and more importantly whether generaldomain text-only SFT can transfer across domains and input modalities.

We start with a cross-distribution investigation where we fine-tune an instruction-tuned VLM, Qwen2.5-VL-7B-Instruct (Bai et al., 2025) using text-only MedQA (Jin et al., 2021), a specialized medical QA dataset. The training signal comprises detailed long-COT reasoning traces, distilled via rejection sampling (Huang et al., 2023; Zelikman et al., 2022) from QwQ-32B (Team, 2025b). We then evaluate the model’s generalization on an out-of-distribution task: MMLU-ProHealth, the healthcare subset of MMLU-Pro (Wang et al., 2024b). The full training and evaluation details are provided in Sections 2.3 and 3. As shown in the top rows of Table 1, all the SFT approaches consistently transfers the improvements from the source task (MedQA) to the target task (MMLU-Pro-Health), demonstrating strong cross-task generalization from text-only SFT.

To further explore the scope of generalization, we perform SFT using OpenThoughts-114k (Team, 2025a), a text-only dataset covering math and science reasoning distilled from DeepSeek-R1 (DeepSeek-AI et al., 2025). Results in Table 1 show the SFT model yields consistent performance gains across three generalization axes: cross-domain (X-domain), cross-modality (X-modality) and the intersection of the two axes. This indicates that SFT with general-domain text-data already shows strong generalization capabilities. In addition, while CoT prompting alone provides some transferable gains, they are notably smaller compared with SFT, indicating that the VLM’s inherent CoT capability prior to SFT is limited and CoT SFT is essential to fully unlock generalizable reasoning.

Role of Reasoning in Generalization To isolate the contribution of reasoning, we compare models trained with and without explicit CoT supervision: (1) a non-CoT SFT model trained directly on input-label pairs, and (2) a CoT SFT model trained with reasoning traces. Across all experiments in Table 1, the CoT-based SFT model consistently outperforms its non-CoT counterpart. Even among

the baseline models, those equipped with CoT prompting outperform those without it, underscoring that explicit reasoning is a prerequisite for high performance on reasoning-centric tasks. Furthermore, we observe that CoT-based models not only perform better in-distribution but also exhibit stronger transfer capabilities. Notably, the performance lift from CoT SFT over the CoT baseline is substantially larger than the corresponding improvement from non-CoT SFT over the non-CoT baseline across all settings. This suggests that explicit reasoning is the key to achieving both the highest in-distribution performance and the optimal generalization across new domains and modalities.

- Table 1: Comparing baseline and text-only SFT with and without CoT in two settings: (1) Training on MedQA and evaluating on in & out of distribution tasks; and (2) Training on general-domain OpenThoughts and evaluating on tasks across modalities and domains.

Training: MedQA Eval: in & out of distribution

Non-CoT CoT Modality Baseline SFT Baseline SFT

MedQA (In distribution) Textual 49.3 57.6 (+8.3) 50.5 61.9 (+11.4) MMLU-Pro-H (Out of dist.) Textual 43.7 47.2 (+3.5) 47.8 54.2 0(+6.4)

Training: OpenThoughts (general-domain) Eval: across tasks, modalities and domains

Non-CoT CoT Modality Baseline SFT Baseline SFT MMLU-Pro (general-domain text) Textual 39.5 39.5 (+0.0) 47.6 50.4 0(+2.8)

MedQA (X-domain) Textual 49.3 52.1 (+2.8) 50.5 55.2 0(+4.7) MMMU-Pro (X-modality) Multimodal 34.6 36.0 (+1.4) 38.3 40.4 0(+2.1)

MMMU-Pro-H (X-modality&domain) Multimodal 28.1 30.2 (+2.1) 34.3 37.9 0(+3.6)

Takeaway 2.1.1

SFT on general-domain text-only data, when enriched with long CoTs, can endow models with generalizable reasoning capabilities that transfer across tasks, domains, and modalities.

Forced-Exiting: Mitigating Endless Thinking in Long-CoT SFT While long-CoT SFT effectively learns transferrable reasoning patterns, such as self-reflection, verification, and correction (see qualitative analyses in Appendix H), we observe a recurring challenge: the model occasionally fails to terminate its responses. Specifically, the model continues generating tokens indefinitely without producing a stop signal. Such endless thinking occurs approximately 17% of generations for both MedQA and MMLU-Pro-Health tasks, as reported in (Table 2). Upon closer inspection, these failure cases are often characterized by verbose, repetitive output, where the model persistently attempts to “re-think” or “double-check”, an artifact likely stemming from the reasoning mechanism introduced during long-CoT SFT (DeepSeek-AI et al., 2025).

To address this, we implement a forced-exiting mechanism inspired by the method introduced in Muennighoff et al. (2025). This involves appending a designated stop token ‘</think>’ once the output sequence reaches a predefined length threshold. This soft intervention encourages the model to wrap up its reasoning process within a reasonable token budget. As shown in Table 2, applying forced-exiting effectively reduces endless thinking, thereby improving final task accuracy.

- Table 2: Forced-exiting mitigates endless thinking and improves final task accuracy from CoT SFT Task Before Forced-Exiting After Forced-Exiting

% endless thinking Accuracy % endless thinking Accuracy

MedQA 17.1% 56.0 0% 61.9 MMLU-Pro-Health 17.3% 49.0 0.1% 54.2

Takeaway 2.1.2

Forced-exiting provides a practical and effective solution to address the endless thinking behavior in long-CoT SFT, improving both reliability and downstream task performance.

#### 2.2 RQ2: Will Mathematical Text-only RL Improve Generalizable Reasoning?

Reinforcement Learning (RL) has become an effective approach for training models to reason. In this section, we explore whether mathematical text-only RL can promote generalizable reasoning.

Learning Algorithm We employ GRPO (Shao et al., 2024) as our RL algorithm (Sutton, 1988). GRPO has recently become favored over algorithms such as PPO (Schulman et al., 2017) as it does not require an additional value function. Instead, GRPO computes the advantage by comparing responses within each query-specific group. The detailed formulation is provided in Appendix A.

To stabilize and improve GRPO training, we incorporate several enhancements inspired by recent work (Liu et al., 2025a; Yu et al., 2025): (1) using a higher upper clipping threshold (clip-higher), which promotes sample diversity and mitigates entropy collapse; (2) employing a token-level policy gradient loss, which normalizes the loss by the total number of tokens within a batch, thus addressing the response-length bias present in GRPO; and (3) reducing or entirely removing the KL penalty, a change beneficial in training models performing long chain-of-thought (CoT) reasoning, where the policy distribution may significantly diverge from the reference model.

Verifiable Rewards Rather than relying on a learned reward model, which often encounters rewardhacking issues (Gao et al., 2022), we directly utilize the final accuracy on verifiable tasks as our reward metric. Specifically, a response receives a reward of r = 1 if it matches the ground truth answer under semantic equivalence (i.e., is equivalent(ˆy,y)), and r = 0 otherwise. We omit format rewards, as our policy model already adheres well to instruction formatting.

The Role of Training Data: Math as a Generalization Anchor A central question in text-only RL is: What type of training data best supports reasoning generalization? While reasoning models are often fine-tuned on domain-specific datasets, we hypothesize that mathematics serves as an especially effective domain for promoting generalizable reasoning, as math tasks naturally elicit long, structured chains of thought that may transfer more effectively across domains.

To test this, we compare RL fine-tuning on two data sources: (1) math textual questions from Orz (Hu et al., 2025), and (2) MedQA, a domain-specific medical textual QA dataset. Results in

- Table 3 reveal a clear asymmetry: models trained on math data not only improve on math benchmarks (e.g., GSM8K, MathVista), but also show strong gains on out-of-domain medical tasks. In contrast, models trained on MedQA excel in medical benchmarks but do not consistently improve math tasks. This suggests that math serves as an “anchor domain”, whose reasoning structure is sufficiently rich and general to support cross-domain&modality transfer. Supporting this, we observe in Table 3 that math-trained models consistently generate longer responses across benchmarks. Furthermore, across all models, responses to math tasks are longer than those to medical tasks, highlighting math’s inherent demand for extended reasoning. That said, domain-specific data remains important. RL fine-tuning on MedQA yields the best performance on medical tasks, indicating that in-domain RL is still critical for achieving optimal performance in specialized domains.

- Table 3: Comparing the effect of the training data domain (math vs medicine) on RL generalization. We report accuracy [response length in number of words] for each experiment.

MMLU-Pro-Health OmniMedVQA GSM8K MathVision (medical text) (medical multimodal) (math text) (math multimodal)

Method

Baseline 47.8 [105] 57.4 0[84] 86.0 [146] 24.7 [293] + text-only RL Math (Orz) 50.7 [166] 58.4 [112] 88.8 [200] 28.1 [580] + text-only RL Medicine (MedQA) 53.6 0[87] 59.2 0[72] 85.7 [135] 26.2 [293]

Takeaway 2.2.1

Mathematics provides a strong foundation for generalizable reasoning in RL, serving as an anchor domain. However, domain-specific data remains essential for optimal performance in specialized contexts.

RL vs SFT Prior work has positioned RL as a powerful yet volatile method for promoting generalizable reasoning (Chu et al., 2025), while SFT is widely recognized for its stability and efficacy in capturing structured reasoning patterns. However, RL is often plagued by training instability and convergence issues, particularly when applied in isolation (Yeo et al., 2025). To better understand their relative strengths, we conduct a direct comparison between RL and SFT in fostering

generalization across task distributional shifts, domain shifts and modality transitions. Specifically, we compare general-domain text-only SFT (trained on general-domain OpenThoughts data) and RL (trained on math questions from Orz), and their combination (SFT + RL). Alongside generaldomain text task (MMLU-Pro), we evaluated the models in three distinct generalization settings: cross-domain (medical text: MMLU-Pro-Health), cross-modality (multimodal: MMMU-Pro), cross domain&modality (multimodal medical: NEJM Image Challenge). In Table 4, we first observe that general-domain SFT and RL can both significantly improve from baseline across all the generalization settings, indicating both SFT and RL can elicit generalizable reasoning. We further notice that pure RL is overall slightly worse than SFT. This performance gap can be attributed to RL’s difficulty in learning long-CoT reasoning from scratch, evidenced by its shorter, less structured outputs. In contrast, SFT excels at acquiring rich CoT patterns through supervised distillation, yielding significantly longer and more coherent responses. Interestingly, when RL is applied after SFT, it benefits from the structured reasoning foundation established by SFT, and further refines the model’s capabilities via RLVR. We also observe that RL + SFT is able to regulate the endless thinking issue from SFT as the percentage of responses that exceeds the maximum length decreases during RL training, as shown in the training response clip ratio in Figure 5. In summary, the hybrid approach, generaldomain text-only SFT + RL, achieves the best results across all generalization settings, effectively combining SFT’s stability and inductive strength with RL’s reward-guided optimization2.

- Table 4: Comparison of performance from general-domain text-based RL, SFT, and SFT + RL in general-domain textual task (MMLU-Pro) and generalization setups including cross-modality (Xmodality), cross-domain (X-domain) and combined cross-domain&modality (X-modality&domain) tasks. SFT is trained with OpenThoughts data and RL is trained with math data.

MMLU-Pro MMLU-Pro-Health MMMU-Pro NEJM Image Challenge Response Len (G-domain text) (X-domain) (X-modality) (X-domain&modality) (#words)

Method

Baseline 47.6 47.8 38.3 41.8 122 SFT 50.4 50.4 40.4 45.0 946 RL 50.3 50.3 40.6 44.4 232 SFT + RL 53.3 53.1 42.5 45.7 977

Takeaway 2.2.2

Combining SFT with RL, both trained on general-domain text-only data, proves to be the most effective strategy for achieving optimal performance and robust generalization.

#### 2.3 Putting it all together: X-REASONER

Our investigation reveals that general-domain text-only post-training, when carefully designed, can drive strong generalizable reasoning across tasks, domains, and modalities. Therefore, we conclude our investigation by consolidating our findings into a coherent training recipe, illustrated in Figure 2, culminating in a powerful generalizable reasoning model named X-REASONER. Specifically, initialized with Qwen2.5-VL-7B-Instruct (Bai et al., 2025), our training recipe follows:

- Step 1: text-only general-domain Long-CoT SFT. We begin by applying SFT to elicit explicit structured reasoning capabilities using long-CoT reasoning traces. We train our model on the general-domain open-thoughts/OpenThoughts-114k dataset (Team, 2025a). This dataset contains curated reasoning traces on 114k math, coding and science questions, distilled by the DeepSeek-R1 model. We fine-tune the model for 4 epochs with a learning rate of 1 × 10−5. i
- Step 2: text-only math RLVR. After SFT, we further refine our model using RL with verifiable rewards to enhance its reasoning accuracy and generalization. We trainin our model on Orz-math-57k, a set of 57k mathematical textual questions curated by Hu et al. (2025). During this stage, we set the total training episodes/epochs to 3, use a learning rate of 3 × 10−6, a global batch size of 128, and sample 8 rollouts per query with a maximum response length of 4,096 tokens.

Detailed hyperparameters for each training stage are provided in Appendix B. The SFT stage was conducted using 8 40GB A100 GPUs over 8 hours, and the RLVR stage utilized 32 40GB A100 GPUs for approximately 56 hours. We provide visualization of training dynamics in Appendix G.

2We also conduct a more controlled comparison between SFT and RL, both trained on the same MedQA data, and arrive at the same conclusion (Table 9).

## 3 A Comprehensive Evaluation of X-REASONER

Evaluation Setup To thoroughly assess the generalizability of X-REASONER, we evaluate its performance across four distinct settings: (1) General-domain text-only tasks; (2) Generaldomain multimodal tasks (X-modality); (3) Specialized-domain text-only tasks (X-domain); and (4) Specialized-domain multimodal tasks (X-modality&domain). Detailed descriptions of these tasks can be found in Appendix D. For inference, we utilize the vLLM backend (Kwon et al., 2023). Consistent with recent findings on reproducibility (Hochlehnert et al., 2025), we observe performance variability even under greedy decoding (i.e., temperature=0), attributed to hardware differences, software stack inconsistencies, runtime environments, etc. In this study, we report greedy decoding results to compare with previous studies. In addition, to ensure the robustness and reproducibility, we report three additional metrics computed over five runs at temperature = 0.3: average accuracy, majority-vote accuracy, and pass@n accuracy. Majority-vote accuracy aligns with the self-consistency method introduced by Wang et al. (2023), which has proven effective for scaling up parallel test-time evaluation. Pass@n accuracy provides further insight into the model’s effective exploration of the answer space, indicating potential performance gains achievable through additional reinforcement learning. Unless otherwise indicated, we consistently use CoT prompting templates for both baselines and X-REASONER evaluations. Complete exact prompt templates are provided in Appendix C. Additionally, we apply a forced-exiting mechanism as described by Muennighoff et al. (2025), capping output generation at 4,096 tokens. This mechanism not only improves memory efficiency but also effectively handles endless thinking cases, as elaborated in Section 2.1.

100

96.4

Baseline (avg@5) Baseline (pass@5)

Baseline (majority@5) X-Reasoner (avg@5)

X-Reasoner (pass@5) X-Reasoner (majority@5)

| |
|---|

| |
|---|

94.2

92.2

| |
|---|

| |
|---|

| |
|---|

89.7

90

[Figure 18]

[Figure 19]

89.3

81.7 82.8

86.0

80.5

80

76.9

75.7

74.6

73.6

69.6

70

Accuracy(%)

67.2

65.6

64.9

63.9

66.8

60.9

64.6

59.2

60

58.2

61.8

56.7

59.4

52.9 50.6

52.2

51.1

54.8

53.3

50

51.3

45.5

47.6

41.0

40

42.5

38.3

31.9

30

26.5

29.4

24.7

20

MMLU-Pro GSM8K MMMU (val) MMMU-Pro MMStar MathVista (testmini)MathVision

Text Benchmarks Multimodal Benchmarks (X-Modality)

- Figure 3: Comparing X-REASONER and baseline on text-only benchmarks and multi-modal benchmarks. Despite being trained with general-domain text-only data, X-REASONER can significantly improve multi-modal benchmarks, showing the generalization of X-REASONER’s reasoning.

#### 3.1 X-REASONER’s Cross-Modality Generalization

In Figure 3, we assess the cross-modality generalization capabilities of X-REASONER. We first validate its effectiveness on text-only tasks, observing significant performance gains across benchmarks such as MMLU-Pro and GSM8K. Crucially, these improvements robustly transfer to multimodal contexts, reflected consistently across average, majority vote, and pass@5 accuracy. Such consistent improvements indicate not only overall superior reasoning capability, but also suggest that X-REASONER has a more effective and expansive search space for further improvements.

Comparisons against SOTA multimodal reasoning models are presented in Table 5. X-REASONER achieves SOTA on MMMU, MMMU-Pro and MathVista, while remaining competitive on others. Notably, these SOTA models are extensively trained on multimodal data, underscoring the critical finding that text-only reasoning training alone is sufficient—and likely accounts for the majority of learning necessary for effective multimodal reasoning.

Qualitative analysis in Appendix H highlights that X-REASONER consistently generates more sophisticated reasoning outputs compared to the baseline. Reasoning patterns demonstrated by XREASONER include detailed planning, verification (e.g., ”wait” steps), and error correction. In the

cross-modality setting, X-REASONER seamlessly integrates visual information into its reasoning chains, providing step-by-step reasoning with verification. For example, in Table 11, X-REASONER methodically reviews all bar values in a plot, avoiding the oversight errors prevalent in the baseline. X-REASONER also identifies and corrects mistakes within its reasoning process. These observations reinforce our conclusion that X-REASONER successfully internalizes generic reasoning patterns through text-only training, effectively extending them across modalities.

- Table 5: X-REASONER outperforms SOTA multimodal models on the 7B/8B scale on challenging multimodal tasks. All results are based on greedy decoding.

Model

Reasoning MMMU MMMU-Pro MathVista MathVision Data Source (Val) (testmini)

Qwen-VL-2.5-7B-Instruct (Bai et al., 2025) - 53.0 38.7 62.8 25.0 R1-Onevision-7B (Yang et al., 2025) multimodal - - 64.1 29.9 MAmmoTH-VL2-7B (Jia et al., 2025) multimodal 54.7 40.7 68.1 MM-Eureka-8B (Meng et al., 2025) multimodal 49.2 - 67.1 22.2 Mulberry-7B (Yao et al., 2024) multimodal 55.0 36.8 63.1 -

X-REASONER (7B) text 56.4 43.0 69.0 29.6

Takeaway 3.2.1

X-REASONER, trained on text-only data, consistently improves multimodal task performance, matching or surpassing prior SOTA trained explicitly with multimodal data.

Ablation Study: Is X-REASONER simply solving the text-solvable examples? A potential limitation in evaluating vision-language models is their tendency to rely on text-based shortcuts, solving tasks without genuinely integrating visual information. To rigorously confirm that the crossmodality generalization observed with X-REASONER is not merely due to improved text-only shortcut solutions, we conduct an ablation experiment. Specifically, we identify and remove text-solvable examples and assess whether performance gains persist afterward.

To identify these text-solvable examples, we mask visual inputs for multimodal tasks previously evaluated in Section 3.1, performing text-only evaluations using both X-REASONER and the baseline Qwen2.5-VL-7B-Instruct. For each model, we sample three independent responses per question and eliminate examples consistently solved through text alone.

- Table 6 summarizes the number of remaining examples after removing text-solvable examples for each task, along with the corresponding model performances. While a substantial number of textsolvable instances were identified, these primarily reflect models’ textual reasoning shortcuts rather than dataset issues (Yue et al., 2024b). Crucially, X-REASONER maintains performance advantages over the baseline after text-only solvable examples are excluded, affirming true multimodal reasoning capabilities. This confirms that X-REASONER effectively utilizes visual context within the reasoning framework acquired through text-only training.

Table 6: Comparing X-REASONER and baseline performance after removing text-solvable examples. The improvement of X-REASONER remains, indicating true multimodal generalizability.

Task #original - #text-solvable = # Baseline X-REASONER

MMMU 900 - 308 = 592 39.3 41.4 MMMU-Pro 1730 - 259 = 1471 33.4 36.4 MathVista (testmini) 1000 - 257 = 743 57.9 60.6 MathVision 3040 - 448 = 2592 18.6 21.8 MMStar 1500 - 259 = 1241 56.6 59.0

Takeaway 3.2.2

X-REASONER achieves authentic multimodal reasoning capabilities, as evidenced by sustained improvements after excluding text-only solvable examples.

#### 3.2 X-REASONER in Medicine: A Cross-Modality and Cross-Domain Study

In this section, we examine the cross-domain and cross-modality transfer capabilities of XREASONER, specifically within medical contexts. While X-REASONER is trained exclusively on general-domain text data, we further probe the benefits of domain-specific adaptation by developing X-REASONER-MED, which extends X-REASONER with additional training using medical text data.

Evaluation Setup Our evaluation includes three text-only medical tasks: MMLU-Pro-Health (the healthcare subset of MMLU-Pro (Wang et al., 2024b), selected following the setup in (Chen et al., 2024a)), MedQA (the original USMLE version), and MedQA (4-ops: the four-choice version). For multimodal medical evaluations, we utilize healthcare subsets of MMMU (Yue et al., 2024a) and MMMU-Pro (Yue et al., 2024b), designated as MMMU-Health and MMMU-Pro-Health respectively. We also include MedXpertQA-MM (Zuo et al., 2025), OmniMedVQA (Hu et al., 2024) and NEJM Image Challenge (New England Journal of Medicine, 2025), which are specifically designed to evaluate medical knowledge and reasoning.

100

Baseline (avg@5) Baseline (pass@5) Baseline (majority)

X-Reasoner (avg@5) X-Reasoner (pass@5)

X-Reasoner (majority@5) X-Reasoner-Med (avg@5)

X-Reasoner-Med (pass@5) X-Reasoner-Med (majority@5)

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

90

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

84.6

82.0

82.0

81.1

80.0 80.9

80

77.8

76.4

74.5

74.1

73.3

72.0

72.0

72.0

69.6

70

69.0

71.1

65.7

65.4

66.0

66.0

65.1

64.3

Accuracy(%)

66.7

63.5

61.2

60.5

63.5

60.0

60.0

62.9

59.6

60

61.8

58.6

61.6

58.3

57.2

57.7

56.0

55.2

58.0

57.9

57.4

56.3

55.8

55.5

51.8

51.7

53.1

50

48.4

48.9

51.0

50.5

47.1

45.7

47.8

43.3

45.7

42.0

39.4

40

41.8

38.5

38.1

40.0

36.8

34.3

30

26.0

24.2

25.9

21.8

23.8

20

21.4

MedQA (4-ops) MedQA MMLU-Pro-H MMMU-Pro-H MMMU-H OmniMedVQA NEJM MedXpertQA-MM

Medical Text Benchmarks (X-Domain)

Medical Multimodal Benchmarks (X-Domain & X-Modality)

- Figure 4: Comparing X-REASONER, X-REASONER-MED and baseline on text and multimodal medical benchmarks. X-REASONER, trained with general-domain text-only data brings consistent improvement across medical tasks. X-REASONER-MED, obtained by continued training of XREASONER on medical text data, further improves medical domain performance.

Results Figure 4 shows X-REASONER demonstrates robust cross-domain and cross-modality generalization capabilities, surpassing baseline models on both textual and multimodal medical tasks. Remarkably, as detailed in Figure 1, X-REASONER surpassing or matching previous SOTA on MedXpertQA-MM and MMMU-Health, despite previous SOTA models being explicitly trained on multimodal medical data. Qualitative assessments in Appendix H further illustrate X-REASONER’s effective integration of visual information and medical domain-specific reasoning. As an example, X-REASONER accurately identifies medical patterns, such as correctly recognizing white matter abnormalities indicative of progressive multifocal leukoencephalopathy in histology slides (Table 12).

Takeaway 3.3.1

X-REASONER substantially enhances medical task performance, confirming its strong capability for cross-domain and cross-modality generalization.

X-REASONER-MED: Enhancing Medical Performance While X-REASONER shows substantial improvements on medical tasks, our analysis in Section 2.2 highlights the critical role of domainspecific data for achieving optimal in-domain performance. This leads us to investigate whether additional medical-domain text data can further enhance X-REASONER’s performance. To explore this, we extend X-REASONER’s training with text-only MedQA data. Specifically, we perform SFT with distilled CoTs from QwQ-32B (Team, 2025b), followed by RLVR using the same MedQA data.

The resulting model, X-REASONER-MED, achieves further improvements across medical benchmarks. As depicted in Figure 4, X-REASONER-MED consistently surpasses X-REASONER in terms of average and majority vote accuracies. Notably, X-REASONER sometimes attains higher pass@n scores, suggesting it explores a broader search space. Conversely, X-REASONER-MED, benefiting from targeted medical-domain fine-tuning, already leverages this search space more effectively but potentially with reduced room for further gains. Nevertheless, as highlighted in Figure 1, XREASONER-MED sets new SOTA performance for both text-only and multimodal medical tasks, validating our hypothesis that combining general text-based reasoning with domain-specific textbased fine-tuning unlocks substantial additional performance gains for specialized domains.

Takeaway 3.3.2

X-REASONER-MED, resulting from continued training of X-REASONER with medical textonly data, achieves further performance enhancements and establishes new SOTA results in text and multimodal medical benchmarks.

## 4 Related Work

In the context of RL, several methods have been proposed for learning multimodal reasoning. Notable examples include VFT (Liu et al., 2025b), MM-Eureka (Meng et al., 2025), Vision-R1 (Huang et al., 2025a), R1-V (Chen et al., 2025), and the framework proposed by Zhou et al. (2025). These works demonstrate how RL can be used to refine reasoning abilities within VLMs. Peng et al. (2025b) present a two-stage RL approach that leverages both text-based and multimodal data, while Deng et al. (2025) explore an iterative training scheme alternating between SFT and RL on multimodal inputs, beginning with distillation of CoT reasoning from image captions.

On the SFT side, reasoning is commonly distilled from captions or responses synthesized by VLMs. Methods such as MaMMOTH (Guo et al., 2024; Jia et al., 2025), Vision-R1 (Huang et al., 2025a), R1-OneVision (Yang et al., 2025), and Llava-CoT (Xu et al., 2024) follow this strategy to infuse reasoning into multimodal models during SFT. An alternative approach involves linking a text-based reasoning model with a vision encoder to incorporate visual grounding (Peng et al., 2025a).

Recently, there has been growing interest in applying reasoning techniques to the medical domain, initially focusing on the text modality (Zhang et al., 2025a; Huang et al., 2025b; Chen et al., 2024a). Subsequently, researchers have begun exploring RL-based training on multimodal medical datasets to enhance multimodal reasoning performance, e.g., Pan et al. (2025) and Lai et al. (2025).

## 5 Discussions and Limitations

While we show X-REASONER effectively equips vision-language models with generalizable reasoning capabilities, we acknowledge several limitations that offer opportunities for future work:

Base Model Constraints Prior studies have emphasized the importance of starting from pretrained base checkpoints, rather than instruction-tuned models, to better incentivize the emergence of reasoning behaviors (DeepSeek-AI et al., 2025; Hu et al., 2025). However, in our case, we are constrained by computational resources and the lack of base VLMs in the recent open-source releases, limiting our ability to empirically validate these claims.

Model Scale and Backbone Diversity Prior work suggests that larger models tend to exhibit stronger reasoning capabilities (DeepSeek-AI et al., 2025). However, due to computational constraints, our model size is limited to 7B parameters. We also focused exclusively on the Qwen-VL series and did not evaluate other publicly available VLMs.

Task Scope Our evaluation primarily targets mathematical questions and multiple-choice questions across general and medical domains. While these benchmarks provide clear, verifiable supervision for assessing reasoning capabilities, they do not capture the full spectrum of real-world reasoning challenges. In particular, we have not tested our approach on open-ended generation, interactive dialogue, or instruction-following scenarios. Understanding how reasoning generalizes to such unconstrained settings remains an important area for future exploration.

## 6 Conclusion

In this work, we introduce X-REASONER, a simple yet effective post-training recipe that enhances the reasoning capabilities of VLMs using only general-domain text-based supervision from a combined SFT + RL strategy. Through extensive experiments, we demonstrate that the reasoning skills acquired via X-REASONER’s text-only training pipeline generalize effectively across modalities and domains, surpassing prior SOTA trained with in-domain multimodal data. We further introduce XREASONER-MED, a domain-specialized variant of X-REASONER trained on medical text which achieves new SOTA on a range of medical benchmarks across both text-only and multimodal settings. Our findings underscore the surprising power of textual supervision for learning transferable reasoning in multimodal models. We leave to future work the exploration of continued in-domain multimodal SFT/RL to further strengthen domain-specific and multimodal reasoning capabilities.

## References

Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-VL Technical Report, 2025. URL http://arxiv.org/abs/2502.13923.

Junying Chen, Zhenyang Cai, Ke Ji, Xidong Wang, Wanlong Liu, Rongsheng Wang, Jianye Hou, and Benyou Wang. HuatuoGPT-o1, Towards Medical Complex Reasoning with LLMs, 2024a. URL http://arxiv.org/abs/2412.18925.

Liang Chen, Lei Li, Haozhe Zhao, Yifan Song, and Vinci. R1-v: Reinforcing super generalization ability in vision-language models with less than $3. https://github.com/Deep-Agent/R1-V,

2025. Accessed: 2025-02-02.

Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Jiaqi Wang, Yu Qiao, Dahua Lin, and Feng Zhao. Are We on the Right Way for Evaluating Large Vision-Language Models? In Proceedings of Conference on Neural Information Processing Systems (NeurIPS), 2024b. URL http://arxiv.org/abs/2403.20330.

Zui Chen, Yezeng Chen, Jiaqi Han, Zhijie Huang, Ji Qi, and Yi Zhou. An Empirical Study of Data Ability Boundary in LLMs’ Math Reasoning, 2024c. URL http://arxiv.org/abs/2403. 00799.

Tianzhe Chu, Yuexiang Zhai, Jihan Yang, Shengbang Tong, Saining Xie, Dale Schuurmans, Quoc V. Le, Sergey Levine, and Yi Ma. SFT Memorizes, RL Generalizes: A Comparative Study of Foundation Model Post-training, 2025. URL http://arxiv.org/abs/2501.17161.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. Training Verifiers to Solve Math Word Problems, 2021. URL http://arxiv.org/ abs/2110.14168.

Noel C. F. Codella, Ying Jin, Shrey Jain, Yu Gu, Ho Hin Lee, Asma Ben Abacha, Alberto Santamaria-Pang, Will Guyman, Naiteek Sangani, Sheng Zhang, Hoifung Poon, Stephanie Hyland, Shruthi Bannur, Javier Alvarez-Valle, Xue Li, John Garrett, Alan McMillan, Gaurav Rajguru, Madhu Maddi, Nilesh Vijayrania, Rehaan Bhimai, Nick Mecklenburg, Rupal Jain, Daniel Holstein, Naveen Gaur, Vijay Aski, Jenq-Neng Hwang, Thomas Lin, Ivan Tarapov, Matthew Lungren, and Mu Wei. MedImageInsight: An Open-Source Embedding Model for General Domain Medical Imaging, 2024. URL http://arxiv.org/abs/2410.06542.

DeepSeek-AI, Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, and 190 additional authors. DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning, 2025. URL http://arxiv.org/ abs/2501.12948.

Yihe Deng, Hritik Bansal, Fan Yin, Nanyun Peng, Wei Wang, and Kai-Wei Chang. OpenVLThinker: An Early Exploration to Complex Vision-Language Reasoning via Iterative Self-Improvement,

2025. URL http://arxiv.org/abs/2503.17352. Leo Gao, John Schulman, and Jacob Hilton. Scaling Laws for Reward Model Overoptimization,

##### 2022. URL http://arxiv.org/abs/2210.10760.

Gemma Team, Morgane Riviere, Shreya Pathak, Pier Giuseppe Sessa, Cassidy Hardin, Surya Bhupatiraju, L´eonard Hussenot, Thomas Mesnard, Bobak Shahriari, Alexandre Ram´e, and 188 additional authors. Gemma 2: Improving Open Language Models at a Practical Size, 2024. URL http://arxiv.org/abs/2408.00118.

Jarvis Guo, Tuney Zheng, Yuelin Bai, Bo Li, Yubo Wang, King Zhu, Yizhi Li, Graham Neubig, Wenhu Chen, and Xiang Yue. MAmmoTH-VL: Eliciting Multimodal Reasoning with Instruction Tuning at Scale, 2024. URL http://arxiv.org/abs/2412.05237.

Andreas Hochlehnert, Hardik Bhatnagar, Vishaal Udandarao, Samuel Albanie, Ameya Prabhu, and Matthias Bethge. A Sober Look at Progress in Language Model Reasoning: Pitfalls and Paths to Reproducibility, 2025. URL http://arxiv.org/abs/2504.07086.

Jingcheng Hu, Yinmin Zhang, Qi Han, Daxin Jiang, Xiangyu Zhang, and Heung-Yeung Shum. Open-reasoner-zero: An open source approach to scaling up reinforcement learning on the base model, 2025. URL https://arxiv.org/abs/2503.24290.

Yutao Hu, Tianbin Li, Quanfeng Lu, Wenqi Shao, Junjun He, Yu Qiao, and Ping Luo. OmniMedVQA: A New Large-Scale Comprehensive Evaluation Benchmark for Medical LVLM. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024. URL http://arxiv.org/abs/2402.09181.

Jiaxin Huang, Shixiang Gu, Le Hou, Yuexin Wu, Xuezhi Wang, Hongkun Yu, and Jiawei Han. Large Language Models Can Self-Improve. In Proceedings of Conference on Empirical Methods in Natural Language Processing (EMNLP), 2023. URL https://aclanthology.org/2023. emnlp-main.67.

Wenxuan Huang, Bohan Jia, Zijie Zhai, Shaosheng Cao, Zheyu Ye, Fei Zhao, Zhe Xu, Yao Hu, and Shaohui Lin. Vision-R1: Incentivizing Reasoning Capability in Multimodal Large Language Models, 2025a. URL http://arxiv.org/abs/2503.06749.

Xiaoke Huang, Juncheng Wu, Hui Liu, Xianfeng Tang, and Yuyin Zhou. m1: Unleash the potential of test-time scaling for medical reasoning with large language models, 2025b. URL https: //arxiv.org/abs/2504.00869.

Yiming Jia, Jiachen Li, Xiang Yue, Bo Li, Ping Nie, Kai Zou, and Wenhu Chen. VisualWebInstruct: Scaling up Multimodal Instruction Data through Web Search, 2025. URL http://arxiv.org/ abs/2503.10582.

Di Jin, Eileen Pan, Nassim Oufattole, Wei-Hung Weng, Hanyi Fang, and Peter Szolovits. What disease does this patient have? a large-scale open domain question answering dataset from medical exams. Applied Sciences, 11(14):6421, 2021.

Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with pagedattention. In Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles, 2023.

Yuxiang Lai, Jike Zhong, Ming Li, Shitian Zhao, and Xiaofeng Yang. Med-R1: Reinforcement Learning for Generalizable Medical Reasoning in Vision-Language Models, 2025. URL http: //arxiv.org/abs/2503.13939.

Chunyuan Li, Cliff Wong, Sheng Zhang, Naoto Usuyama, Haotian Liu, Jianwei Yang, Tristan Naumann, Hoifung Poon, and Jianfeng Gao. Llava-med: Training a large language-and-vision assistant for biomedicine in one day. In A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine (eds.), Advances in Neural Information Processing Systems, volume 36, pp. 28541–28564. Curran Associates,

Inc., 2023. URL https://proceedings.neurips.cc/paper_files/paper/2023/file/ 5abcdf8ecdcacba028c6662789194572-Paper-Datasets_and_Benchmarks.pdf.

Zichen Liu, Changyu Chen, Wenjun Li, Penghui Qi, Tianyu Pang, Chao Du, Wee Sun Lee, and Min Lin. Understanding R1-Zero-Like Training: A Critical Perspective, 2025a. URL http: //arxiv.org/abs/2503.20783.

Ziyu Liu, Zeyi Sun, Yuhang Zang, Xiaoyi Dong, Yuhang Cao, Haodong Duan, Dahua Lin, and Jiaqi Wang. Visual-rft: Visual reinforcement fine-tuning. arXiv preprint arXiv:2503.01785, 2025b.

Ilya Loshchilov and Frank Hutter. Decoupled Weight Decay Regularization. In Proceedings of International Conference on Learning Representations (ICLR), 2019. URL http://arxiv.org/ abs/1711.05101.

Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, KaiWei Chang, Michel Galley, and Jianfeng Gao. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. In The Twelfth International Conference on Learning Representations, 2024.

Renqian Luo, Liai Sun, Yingce Xia, Tao Qin, Sheng Zhang, Hoifung Poon, and Tie-Yan Liu. Biogpt: generative pre-trained transformer for biomedical text generation and mining. Briefings in bioinformatics, 23(6):bbac409, 2022.

Fanqing Meng, Lingxiao Du, Zongkai Liu, Zhixiang Zhou, Quanfeng Lu, Daocheng Fu, Botian Shi, Wenhai Wang, Junjun He, Kaipeng Zhang, et al. Mm-eureka: Exploring visual aha moment with rule-based large-scale reinforcement learning. arXiv preprint arXiv:2503.07365, 2025.

Niklas Muennighoff, Zitong Yang, Weijia Shi, Xiang Lisa Li, Li Fei-Fei, Hannaneh Hajishirzi, Luke Zettlemoyer, Percy Liang, Emmanuel Cand`es, and Tatsunori Hashimoto. S1: Simple test-time scaling, 2025. URL http://arxiv.org/abs/2501.19393.

Robert Osazuwa Ness, Katie Matton, Hayden Helm, Sheng Zhang, Junaid Bajwa, Carey E. Priebe, and Eric Horvitz. Medfuzz: Exploring the robustness of large language models in medical question answering, 2024. URL https://arxiv.org/abs/2406.06573.

New England Journal of Medicine. Image challenge, 2025. URL https://www.nejm.org/

image-challenge. Accessed: 2025-05-02.

Harsha Nori, Yin Tat Lee, Sheng Zhang, Dean Carignan, Richard Edgar, Nicolo Fusi, Nicholas King, Jonathan Larson, Yuanzhi Li, Weishung Liu, Renqian Luo, Scott Mayer McKinney, Robert Osazuwa Ness, Hoifung Poon, Tao Qin, Naoto Usuyama, Chris White, and Eric Horvitz. Can Generalist Foundation Models Outcompete Special-Purpose Tuning? Case Study in Medicine, 2023. URL http://arxiv.org/abs/2311.16452.

Harsha Nori, Naoto Usuyama, Nicholas King, Scott Mayer McKinney, Xavier Fernandes, Sheng Zhang, and Eric Horvitz. From Medprompt to o1: Exploration of Run-Time Strategies for Medical Challenge Problems and Beyond, 2024. URL http://arxiv.org/abs/2411.03590.

OpenAI. Openai o1 system card, 2024. URL https://arxiv.org/abs/2412.16720. OpenAI. Openai o3 and o4-mini system card. https://cdn.openai.com/pdf/

2221c875-02dc-4789-800b-e7758f3722c1/o3-and-o4-mini-system-card.pdf, 2025. Accessed: 2025-04-29.

Pallets. Jinja. https://github.com/pallets/jinja, 2024.

Jiazhen Pan, Che Liu, Junde Wu, Fenglin Liu, Jiayuan Zhu, Hongwei Bran Li, Chen Chen, Cheng Ouyang, and Daniel Rueckert. MedVLM-R1: Incentivizing Medical Reasoning Capability of Vision-Language Models (VLMs) via Reinforcement Learning, 2025. URL http://arxiv. org/abs/2502.19634.

Yi Peng, Chris, Xiaokun Wang, Yichen Wei, Jiangbo Pei, Weijie Qiu, Ai Jian, Yunzhuo Hao, Jiachun Pan, Tianyidan Xie, Li Ge, Rongxian Zhuang, Xuchen Song, Yang Liu, and Yahui Zhou. Skywork r1v: Pioneering multimodal reasoning with chain-of-thought. https://github.com/SkyworkAI/Skywork-R1V/blob/main/report/Skywork R1V.pdf, 2025a. URL https://huggingface.co/Skywork/Skywork-R1V-38B.

Yingzhe Peng, Gongrui Zhang, Miaosen Zhang, Zhiyuan You, Jie Liu, Qipeng Zhu, Kai Yang, Xingzhong Xu, Xin Geng, and Xu Yang. Lmm-r1: Empowering 3b lmms with strong reasoning abilities through two-stage rule-based rl. arXiv preprint arXiv:2503.07536, 2025b.

Hoifung Poon, Tristan Naumann, Sheng Zhang, and Javier Gonz´alez Hern´andez. Precision health in the age of large language models. In Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, KDD ’23, pp. 5825–5826, New York, NY, USA, 2023. Association for Computing Machinery. ISBN 9798400701030. doi: 10.1145/3580305.3599568. URL https://doi.org/10.1145/3580305.3599568.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal Policy Optimization Algorithms, 2017. URL http://arxiv.org/abs/1707.06347.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models, 2024. URL http://arxiv.org/abs/2402. 03300.

Yanzhou Su, Tianbin Li, Jiyao Liu, Chenglong Ma, Junzhi Ning, Cheng Tang, Sibo Ju, Jin Ye, Pengcheng Chen, Ming Hu, Shixiang Tang, Lihao Liu, Bin Fu, Wenqi Shao, Xiaowei Hu, Xiangwen Liao, Yuanfeng Ji, and Junjun He. GMAI-VL-R1: Harnessing Reinforcement Learning for Multimodal Medical Reasoning, 2025. URL http://arxiv.org/abs/2504.01886.

Richard S. Sutton. Learning to predict by the methods of temporal differences. Machine Learning,

3:9–44, 1988. URL http://link.springer.com/10.1007/BF00115009. OpenThoughts Team. Open Thoughts. https://open-thoughts.ai, January 2025a. Qwen Team. Qwq-32b: Embracing the power of reinforcement learning, March 2025b. URL

##### https://qwenlm.github.io/blog/qwq-32b/.

Haozhe Wang, Chao Qu, Zuming Huang, Wei Chu, Fangzhen Lin, and Wenhu Chen. Vlrethinker: Incentivizing self-reflection of vision-language models with reinforcement learning. arXiv preprint arXiv:2504.08837, 2025.

Ke Wang, Junting Pan, Weikang Shi, Zimu Lu, Houxing Ren, Aojun Zhou, Mingjie Zhan, and Hongsheng Li. Measuring multimodal mathematical reasoning with math-vision dataset. In The Thirty-eight Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2024a. URL https://openreview.net/forum?id=QWTCcxMpPA.

Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. Self-Consistency Improves Chain of Thought Reasoning in Language Models. In Proceedings of International Conference on Learning Representations (ICLR), 2023. URL http://arxiv.org/abs/2203.11171.

Yubo Wang, Xueguang Ma, Ge Zhang, Yuansheng Ni, Abhranil Chandra, Shiguang Guo, Weiming Ren, Aaran Arulraj, Xuan He, Ziyan Jiang, et al. Mmlu-pro: A more robust and challenging multitask language understanding benchmark. In The Thirty-eight Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2024b.

Yiqing Xie, Sheng Zhang, Hao Cheng, Pengfei Liu, Zelalem Gero, Cliff Wong, Tristan Naumann, Hoifung Poon, and Carolyn Rose. Doclens: Multi-aspect fine-grained evaluation for medical text generation. arXiv preprint arXiv:2311.09581, 2023.

Guowei Xu, Peng Jin, Li Hao, Yibing Song, Lichao Sun, and Li Yuan. LLaVA-o1: Let Vision Language Models Reason Step-by-Step, 2024. URL http://arxiv.org/abs/2411.10440.

Yi Yang, Xiaoxuan He, Hongkun Pan, Xiyan Jiang, Yan Deng, Xingtao Yang, Haoyu Lu, Dacheng Yin, Fengyun Rao, Minfeng Zhu, et al. R1-onevision: Advancing generalized multimodal reasoning through cross-modal formalization. arXiv preprint arXiv:2503.10615, 2025.

Huanjin Yao, Jiaxing Huang, Wenhao Wu, Jingyi Zhang, Yibo Wang, Shunyu Liu, Yingjie Wang, Yuxin Song, Haocheng Feng, Li Shen, et al. Mulberry: Empowering mllm with o1-like reasoning and reflection via collective monte carlo tree search. arXiv preprint arXiv:2412.18319, 2024.

Edward Yeo, Yuxuan Tong, Morry Niu, Graham Neubig, and Xiang Yue. Demystifying long chainof-thought reasoning in llms. 2025. URL https://arxiv. org/pdf/2502.03373, 2025.

Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Tiantian Fan, Gaohong Liu, Lingjun Liu, Xin Liu, Haibin Lin, Zhiqi Lin, Bole Ma, Guangming Sheng, Yuxuan Tong, Chi Zhang, Mofan Zhang, Wang Zhang, Hang Zhu, Jinhua Zhu, Jiaze Chen, Jiangjie Chen, Chengyi Wang, Hongli Yu, Weinan Dai, Yuxuan Song, Xiangpeng Wei, Hao Zhou, Jingjing Liu, Wei-Ying Ma, Ya-Qin Zhang, Lin Yan, Mu Qiao, Yonghui Wu, and Mingxuan Wang. DAPO: An OpenSource LLM Reinforcement Learning System at Scale, 2025. URL http://arxiv.org/abs/ 2503.14476.

Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, et al. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 9556–9567, 2024a.

Xiang Yue, Tianyu Zheng, Yuansheng Ni, Yubo Wang, Kai Zhang, Shengbang Tong, Yuxuan Sun, Botao Yu, Ge Zhang, Huan Sun, et al. Mmmu-pro: A more robust multi-discipline multimodal understanding benchmark. arXiv preprint arXiv:2409.02813, 2024b.

Juan Manuel Zambrano Chaves, Shih-Cheng Huang, Yanbo Xu, Hanwen Xu, Naoto Usuyama, Sheng Zhang, Fei Wang, Yujia Xie, Mahmoud Khademi, Ziyi Yang, Hany Awadalla, Julia Gong, Houdong Hu, Jianwei Yang, Chunyuan Li, Jianfeng Gao, Yu Gu, Cliff Wong, Mu Wei, Tristan Naumann, Muhao Chen, Matthew P. Lungren, Akshay Chaudhari, Serena Yeung-Levy, Curtis P. Langlotz, Sheng Wang, and Hoifung Poon. A clinically accessible small multimodal radiology model and evaluation metric for chest x-ray findings. Nature Communications, 16(1), April 2025. ISSN 2041-1723. doi: 10.1038/s41467-025-58344-x. URL http://dx.doi.org/10.1038/ s41467-025-58344-x.

Eric Zelikman, Yuhuai Wu, Jesse Mu, and Noah D. Goodman. STaR: Bootstrapping Reasoning With Reasoning. In Proceedings of Conference on Neural Information Processing Systems (NeurIPS),

##### 2022. URL http://arxiv.org/abs/2203.14465.

Sheng Zhang, Qianchu Liu, Guanghui Qin, Tristan Naumann, and Hoifung Poon. Med-RLVR: Emerging Medical Reasoning from a 3B base model via reinforcement Learning, 2025a. URL http://arxiv.org/abs/2502.19655.

Sheng Zhang, Yanbo Xu, Naoto Usuyama, Hanwen Xu, Jaspreet Bagga, Robert Tinn, Sam Preston, Rajesh Rao, Mu Wei, Naveen Valluri, et al. A multimodal biomedical foundation model trained from fifteen million image–text pairs. NEJM AI, 2(1):AIoa2400640, 2025b.

Hengguang Zhou, Xirui Li, Ruochen Wang, Minhao Cheng, Tianyi Zhou, and Cho-Jui Hsieh. R1zero’s ”aha moment” in visual reasoning on a 2b non-sft model. arXiv preprint arXiv:2503.05132, 2025.

Yuxin Zuo, Shang Qu, Yifei Li, Zhangren Chen, Xuekai Zhu, Ermo Hua, Kaiyan Zhang, Ning Ding, and Bowen Zhou. MedXpertQA: Benchmarking Expert-Level Medical Reasoning and Understanding, 2025. URL http://arxiv.org/abs/2501.18362.

## A GRPO

Let πθ be a vision language model (VLM) based on decoder-only transformers and parameterized by θ. It takes as input a sequence of tokens, denoted by q, such as a question, and autoregressively

decodes the response, denoted by o. Optionally, πθ can also take visual features, such as pictures, as input that is tokenized by a vision encoder. However, visual features are not included in our training paradigm, and q during training is supposed to be natural languages only. We use text-only math datasets for training. Let D be a dataset and (q,a) ∈ D be pairs of questions and answers.

In Group Relative Policy Optimization (Shao et al., 2024, GRPO), for a given query q, the policy model samples a group of responses, and the token-level advantage Aˆi,t for the i-th response is estimated by normalizing its reward ri relative to the group rewards r:

ri − mean(r) std(r)

Aˆi,t =

, (1)

Similar to PPO (Schulman et al., 2017), GRPO employs a clipped surrogate objective but includes a KL-divergence penalty term directly in its loss function to encourage stability:

|oi|

G

πθ(oi,t | q,oi,<t) πθ

1 |oi|

1 G

Aˆi,t,

JGRPO(θ) = Eq∼Q,{o

min

i}Gi=1∼πθold(·|q)

(oi,t | q,oi,<t)

old

t=1

i=1

πθ(oi,t | q,oi,<t) πθ

Aˆi,t,1 − ϵ,1 + ϵ − βDKL [πθ∥πref] , (2)

clip

(oi,t | q,oi,<t)

old

where πθ is the policy model, i.e., VLM. For a query q, we sample G outputs {o1,...,oG} from the old policy model πθ

. Clip ratio ϵ and β are hyper-parameters and πref is the reference VLM.

old

### B Hyperparameters In this section, we discuss the hyperparameters used in our SFT and RL experiments.

- B.1 Supervised Fine-Tuning

For the SFT experiments, we use the AdamW optimizer (Loshchilov & Hutter, 2019) with a learning rate of 1×10−5, weight decay of 0.0, β1 = 0.9 and β2 = 0.999. We adopt a multiplicative learning rate scheduler with a decay rate of 0.8 for each epoch. The training batch size is set as 4 per device and we used 32 GPUs in total. The context length is truncated to 4096 tokens for each sequence. Each dataset will be trained for 4 epochs.

- B.2 Reinforcement Learning

Throughout our experiments, we set KL coefficient β = 1.0 × 10−2 and the clip ratio ϵ = 0.2. For each example, we sample 8 responses with a maximum length of 4096 tokens and sampling temperature of 1.0.

When updating the actor model, we use AdamW (Loshchilov & Hutter, 2019) and set the learning rate as 3.0 × 10−6 and weight decay as 1.0 × 10−2. A warm-up learning rate scheduler is used by using 10% of the total training steps. Gradient norm is applied with a threshold of 1.0. A global batch size (calculated across all devices) is set as 128. All the training are stopped on the convergence of the reward value on the training set.

### C Prompt Templates Below are the prompt templates in Jinja format Pallets (2024) used in training and evaluation:

Prompt Template for Training

You will solve a problem/request. You should provide your thoughts within <think> </think> tags before providing the answer.\nWrite your final answer within <answer> </answer> tags. Here is the question: {{ question }}{%

Prompt Template for Evaluation Multi-choice Questions

You should provide your thoughts within <think> </think> tags, then answer with just one of the options below within <answer> </answer> tags (For example, if the question is \n’Is the earth flat?\n A: Yes \nB: No’, you should answer with <think>...</think> <answer>B: No </answer>). Here is the question: {{ question }}{% Options:\n\n{{ options }}{%

Prompt Template for Evaluation Mathematical Questions

A conversation between User and Assistant. The user asks a question, and you as the assistant solves it. You should first think about the reasoning process in the mind and then provide the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> the final answer as the option letter or the number depending on the question </answer> (For example, if the question is \n’Is the earth flat?\n A: Yes \nB: No’, you should answer with <think> your reasoning </think> <answer>B: No</answer>. If the question is ’What is 1+1?’, you should answer with <think> your reasoning </think> <answer>2 </answer>).\n\nHere is the question: {{ question }}{% Options:\n\n{{ options }}{%

Table 7: Tasks for different evaluation settings.

Evaluation setting Task name Shorthand Used Data size General-domain text-only

GSM8K main (Cobbe et al., 2021) GSM8K 1,319 MMLU-Pro (Wang et al., 2024b) MMLU-Pro 12,032

MMMU (val) (Yue et al., 2024a) MMMU 900 MMMU-Pro (Yue et al., 2024b) MMMU-Pro 1,730 MMStar (Chen et al., 2024b) MMStar 1,500 MathVista (testmini) (Lu et al., 2024) MathVista 1,000 MathVision (Wang et al., 2024a) MathVision 3,040

General-domain multimodal (X-modality)

MedQA (Jin et al., 2021) MedQA 1273 MedQA (4 options) (Jin et al., 2021) MedQA (4-ops) 1,273 MMLU-Pro-Health (Wang et al., 2024b) MMLU-Pro-H 818

Specialized-domain text-only (X-domain)

MMMU-Health (Yue et al., 2024a) MMMU-H 150 MMMU-Pro-Health (Yue et al., 2024b) MMMU-Pro-H 286 MedXpertQA-MM (Chen et al., 2024b) MedXpertQA-MM 2,000 OmniMedVQA (Hu et al., 2024) OmniMedVQA 1,000 NEJM Image Challenge NEJM 947 (New England Journal of Medicine, 2025)

Specialized-domain multimodal (X-modality & X-domain)

## D Evaluation Tasks

Table 7 summarizes tasks used for different evaluation settings in our experiments: general-domain text-only, general-domain multimodal, specialized-domain text-only, and specialized-domain mul-

timodal. We focus specifically on the medical domain as our representative specialized domain due to its significant practical importance and recent intensive research activities (Li et al., 2023; Zambrano Chaves et al., 2025; Zhang et al., 2025b; Luo et al., 2022; Nori et al., 2023, 2024; Codella et al., 2024; Poon et al., 2023; Xie et al., 2023; Ness et al., 2024).

## E Previous SOTA Results

Table 8: Previous SOTA results. * indicates our replicated results.

Task Previous SOTA Model Result

GSM8K MMOS-DeepSeekMath-7B (Chen et al., 2024c) 87.2 MMLU-Pro Gemma-2-9B-it (Gemma Team et al., 2024) 52.1 MMMU-Pro MAmmoTH-VL2-7B (Jia et al., 2025) 40.7

MMMU (Val) Mulberry-7B (Yao et al., 2024) 55.0 MedQA (4-ops) HuatuoGPT-o1-7b (Chen et al., 2024a) 71.6*

MMLU-Pro-H HuatuoGPT-o1-7B (Chen et al., 2024a) 54.3* MMMU-H GMAI-VL-RL (Su et al., 2025) 57.3

MedXpertQA-MM GMAI-VL-Rl (Su et al., 2025) 23.8

## F Comparison of RL and SFT on MedQA data

We conduct a direct comparison between Rl, SFT and their combination with the same MedQA data. As shown in Table 9, pure RL performs worse than SFT and the hybrid SFT + RL performs the best.

- Table 9: Comparison of performance from RL, SFT, and SFT + RL in both in-distribution (ind. text) and generalization settings, including out-of-distribution (ood. text) and cross-modality (Xmodality). All models are trained on MedQA data.

MedQA MMLU-Pro-Health MMMU-Pro-Health Average Response Length (ind. text) (ood. text) (X-modality) (number of words)

Method

baseline 50.5 47.8 34.3 111 SFT 61.9 54.2 39.2 950 RL 57.8 53.6 35.8 103 SFT + RL 64.9 55.8 39.7 1039

## G Training Dynamics

- Figure 5 shows the RL training dynamics when X-REASONER is being trained on Orz (Hu et al.,

2025) and when X-REASONER-MED is being trained on MedQA (Jin et al., 2021), including training reward, policy gradient loss, mean advantage, mean training response length, training response clip ratio, validation accuracy.

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

Figure 5: RL training dynamics of X-REASONER and X-REASONER-MED

## H Qualitative Analysis

- Table 10 shows some example reasoning output from X-REASONER as compared with the baseline. We can clearly see that X-REASONER is able to engage in more complex reasoning traces including detailed planning, verification (”wait). This complex reasoning patterns is successfully transferred to cross-modality setups where X-REASONER seamlessly integrates the vision information into the reasoning chain, and provides detailed step-by-step reasoning with verification for review/correc-

tion. In the example, X-REASONER provides a detailed plan to go over all the bar values whereas the baseline misses half of the bar values. X-REASONER is also able to correct its mistake in the reasoning chain. In the cross-modal cross-modality setting, X-REASONER connects the dots from both visual information and domain knowledge to the reasoning chain, and successfully identifies the patterns (white matter) from the histology slide and refers to its medical domain knowledge to connect it to progressive multifocal leukoencephalopathy.

Table 10: Reasoning output comparison between the baseline Qwen2.5-VL-7B-Instruct model and X-REASONER on an text-only question.

Question The great dragon, Perg, sat ...within a distance of 1000 feet. Polly could throw the gold javelin...for a distance of 400 feet, well within the reach of the dragon’s flames. But when Polly held the sapphire gemstone, she could throw the javelin three times farther than when not holding the gemstone. If holding the gemstone, how far outside of the reach of the dragon’s flames could Polly stand and still hit the dragon with the gold javelin? (from gsm8k main)

###### Response

Baseline <think> To solve this problem, we need to determine the maximum distance Polly can stand from the dragon and still hit it with the javelin when holding the sapphire gemstone. Here’s the step-by-step reasoning: 1. When Polly is not holding the gemstone, she can throw the javelin for a distance of 400 feet. 2. When Polly holds the gemstone, she can throw the javelin three times farther than when not holding the gemstone. Therefore, the distance she can throw the javelin when holding the gemstone is \( 400 \times 3 = 1200 \) feet...</think> <answer>1200</answer> ✗

X-REASONER <think>Okay, let’s see. First, let’s break down the information. Without the gemstone, the javelin’s distance is 400 feet. With the gemstone, it’s three times that. So, 400 feet multiplied by 3 gives the new throwing distance. Let me calculate that: 400 * 3 = 1200 feet... Wait, but is that correct? Let me check again. Wait, the problem says ”how far outside of the reach of the dragon’s flames could Polly stand and still hit the dragon with the gold javelin?” So, if the dragon’s flames reach 1000 feet, then the maximum distance Polly can stand from the dragon’s position and still hit it is 1200 feet. But the question is asking how far outside the dragon’s flames she can stand. So, 1200 feet minus 1000 feet is 200 feet. Therefore, the answer should be 200 feet. </think> <answer>200</answer> ✓

and X-REASONER on a multimodal question.

[Figure 37]

Question How many bars have value below 40? (from MathVista) Response Baseline <think>To determine how many bars have a value below 40, I’ll examine each bar’s

value:

- 1. 2014: 63 (above 40)
- 2. 2015: 95.5 (above 40)
- 3. 2016: 82.2 (above 40)
- 4. 2017: 72.9 (above 40)
- 5. 2018: 58 (above 40)
- 6. 2019: 55.2 (above 40) None of the bars have a value below 40.</think><answer>0</answer> ✗

X-REASONER <think> Okay, let’s see. The question is asking how many bars have a value below 40. The image is a bar chart with two categories: Banknotes and Coins, and years from 2014 to 2019. Each year has two bars, one for each category. The values are given in thousands, so we need to check each bar’s value and count those below 40 thousand. First, let’s go through each year:

2014:

- - Banknotes: 63 thousand. That’s above 40.
- - Coins: 46 thousand. Also above 40.

... Now, let’s count the bars below 40:

- - 2016: Coins (33)
- - 2017: Coins (32.5)
- - 2018: Coins (33.1)
- - 2019: Coins (42.1) - Wait, 42.1 is above 40. Wait, hold on. Wait, 2019’s Coins is 42.1 thousand. 42.1 is above 40. So that’s not below 40. So only the 2016 and 2017 Coins are below 40... Let’s recast:

- 2014: Banknotes 63 (no), Coins 46 (no) → 0
- 2015: Banknotes 95.5 (no), Coins 46 (no) → 0
- 2016: Banknotes 82.2 (no), Coins 33 (yes) → 1
- 2017: Banknotes 72.9 (no), Coins 32.5 (yes) → 1
- 2018: Banknotes 58 (no), Coins 33.1 (yes) → 1
- 2019: Banknotes 55.2 (no), Coins 42.1 (no) → 0

... So total bars below 40: 3. Therefore, the answer is 3. </think> <answer>3</answer> ✓

and X-REASONER on an medical multimodal question.

[Figure 38]

Question What is in the image? A. Progressive Multifocal Leukoencephalopathy (PML) B. Rabies C. West Nile Virus D. Herpes Simplex encephalitis E. Encephalitis Lethargica F. HIV encephalitis G. Lyme Disease H. Syphilis I. Meningitis J. Creutzfeldt-Jakob disease (from MMMU PRO HEALTH)

###### Response

Baseline <think>The image appears to be a histological section, likely from a brain biopsy, given the presence of cellular structures and the context of the question. The pattern of the lesions and the cellular infiltration are characteristic of encephalitis. Herpes Simplex encephalitis (HSE) is a well-known cause of encephalitis, often presenting with a characteristic pattern of perivascular inflammation and necrosis, which can be seen in such images. The lesions in the image seem to have a focal nature, which is consistent with HSE. However, without further clinical information and a more detailed examination, a definitive diagnosis cannot be made. Given the options, the most likely answer is: D. Herpes Simplex encephalitis</think><answer>D. Herpes Simplex encephalitis</answer> ✗

X-REASONER <think> Okay, let’s try to figure out this question. The image shows a histological section with some lesions. The question is about which neurological disease is depicted. Let me go through the options one by one. First, I remember that PML (Progressive Multifocal Leukoencephalopathy) is caused by the JC virus and typically affects the white matter of the brain, leading to lesions that can be seen on MRI as areas of low signal intensity on T2-weighted images. The histology might show demyelination and gliosis. The image here shows some irregular areas with a pinkish hue, which could be consistent with demyelination. But I need to check if the lesions are in the white matter. The image doesn’t show the entire brain, but the lesions appear to be in the white matter, which is typical for PML. Rabies usually presents with encephalitis, but the histology isn’t typically as specific as PML... Encephalitis Lethargica is an old disease, but its histology isn’t as specific as PML. Given the appearance of the lesions in the white matter, the pinkish areas, and the possibility of demyelination, PML seems the most likely. So the answer is A. </think> <answer>A</answer> ✓

