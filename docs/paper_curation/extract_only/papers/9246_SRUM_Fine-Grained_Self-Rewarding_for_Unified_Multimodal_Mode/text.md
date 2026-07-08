# arXiv:2510.12784v2[cs.CV]28Jun2026

## SRUM: Fine-Grained Self-Rewarding for Unified Multimodal Models

Weiyang Jin1,2,5*, Yuwei Niu3*, Jiaqi Liao2, Chengqi Duan1,2, Aoxue Li4, Shenghua Gao2, and Xihui Liu1,2,5†

1 HKU MMLab 2 The University of Hong Kong 3 Peking University 4 Noah’s Ark Lab, Huawei 5 Shenzhen Loop Area Institute weiyangjin@connect.hku.hk

Abstract. Recently, remarkable progress has been made in Unified Multimodal Models (UMMs), which integrate vision-language generation and understanding capabilities within a single framework. However, a model’s strong visual understanding often fails to transfer to visual generation: it may correctly judge prompt-image alignment while failing to generate a faithful image from the same prompt. This raises a compelling question: Can a model improve itself by using its understanding module to reward its generation module? We introduce SRUM, a self-rewarding post-training framework directly applicable to existing UMMs of various designs. SRUM creates a feedback loop where the model’s own understanding module acts as an internal “evaluator”, providing corrective signals to improve generation without additional human-labeled data or external reward models. To provide comprehensive feedback, SRUM uses a global-local dual reward system: a global reward ensures overall visual semantics and layout, while a local reward refines fine-grained, object-level fidelity. SRUM shows strong generalization, boosting performance on T2I-CompBench from 82.18 to 88.37 and on T2I-ReasonBench from 43.82 to 46.75. Overall, our work establishes a powerful paradigm for enabling a UMM’s understanding module to guide and enhance its own generation via self-rewarding.

Keywords: Unified Multimodal Models · Self-improvement · Post-training

### 1 Introduction

Text-to-Image (T2I) models have achieved remarkable progress in generating high-quality and diverse images from given prompts [51,55,57]. However, they often fail to accurately interpret instructions involving world knowledge, complex spatial relationships, detailed attribute binding, or compositional reasoning [28].

*Equal contribution. †Corresponding author: xihuiliu@eee.hku.hk. Project page: https://waynejin0918.github.io/srum_web/.

These limitations point to a fundamental lack of deep semantic understanding in standard T2I models. To address this challenge, researchers have developed Unified Multimodal Models (UMMs), which integrate both understanding and generation capabilities within a single framework [12, 79, 80, 82]. By sharing a common backbone, UMMs possess the inherent potential for synergy, offering a promising path to resolve the comprehension challenges that plague traditional T2I models.

Despite their advanced architecture, a fundamental paradox plagues current UMMs: their capacity to generate falls far behind their ability to understand [8,50,68,74,83]. For instance, a model can often correctly judge the alignment between a detailed prompt and a complex image, yet be incapable of generating a faithful image from that same prompt (Figure 1). This persistent gap between understanding and generation suggests that the key to unlocking better generation lies within the model itself.

To address this challenge, we propose bridging this module gap through self-rewarding. We introduce Self-Rewarding for Unified Multimodal Models (SRUM), a novel post-training framework designed to create a synergistic feedback loop within the model itself. Our core insight is that the solution lies within the UMMs’ own architecture. By treating the generation module as a “student” and the more capable understanding module as an internal “teacher” or “evaluator,” we establish a practical self-rewarding system for improvement, without external reward models, human labels, or additional image data during the training phase.

Furthermore, to effectively guide the generation of complex scenes, a reward signal should provide multi-scale feedback. As our ablation studies confirm, a single, holistic score is insufficient because it fails to provide the fine-grained corrective signals needed for detailed improvement. Therefore, we propose a global-local dual reward framework. The global reward evaluates high-level compositional coherence to ensure overall scene plausibility. Concurrently, the local reward targets object-level details, optimizing attribute binding and spatial arrangements. This synergistic design enables SRUM to enhance the performance of the base model on complex generation tasks.

Through extensive experiments, we demonstrate that our approach significantly improves the composition, reasoning, and visual fidelity of UMMs, showing strong generalization across in-domain and out-of-domain settings. SRUM achieves SOTA results on T2I-CompBench and T2I-ReasonBench, improving the overall score of a strong baseline model from 82.18 to 88.37 in composition and from 43.82 to 46.75 in reasoning. Our key contributions can be summarized as follows:

- – We are the first to propose a comprehensive self-rewarding framework for UMMs at the post-training stage, successfully bridging the gap between their understanding and generation.
- – We introduce a novel dual reward design that combines global compositional assessment with local object-level feedback, providing solid and multi-scale guidance during model training.

Gap between

“Holiday celebrating the birth of Jesus Christ” “give me image about the final exam was a piece of cake” “Generate image of seven books”

Generation & Understanding

Red banana and yellow apple, with the red fruit on top of the yellow fruit.

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

###### UMMs

Generation

###### UMMs

Generation

###### Local Reward Global Reward

###### UMMs

#### SRUM

Understanding

[Figure 5]

Guides Training

###### UMMs

bridging the gap

Understanding

[Figure 6]

[Figure 7]

[Figure 8]

###### UMMs

It shows the Red banana is on top of yellow apple.

Generation

Commonsense Knowledge

Reasoning-informed Generation

Compositional Generation

Correct Wrong

- Fig. 1: The example on the left suggests that the current UMMs’ understanding module has exceeded the capability of its generation module: the generation module is prone to producing incorrect candidate images based on a given prompt in relevant scenarios, a situation which the understanding module can reasonably identify. This not only highlights a gap between understanding and generation but also reveals the potential for understanding to guide generation. Inspired by this insight, we propose SRUM to bridge this gap, particularly in complex generation domains.

– We achieve better performance on complex compositional generation and demonstrate strong generalization. Ultimately, SRUM establishes a powerful paradigm for a UMM’s understanding module to guide its own generation module toward self-improvement.

### 2 Related Works

##### 2.1 Architectures for Unified Multimodal Models

Unified Multimodal Models (UMMs) have emerged as a prominent research direction, aiming to integrate diverse tasks like visual understanding and generation within a single, end-to-end trained architecture. This consolidation seeks to foster synergy across modalities and reduce systemic complexity. Recent architectural paradigms can be broadly categorized. The Purely Autoregressive (AR) approach extends the next-token prediction paradigm of LLMs to visual data, treating images as a sequence of discrete tokens [66,74]. A key refinement in this area involves decoupling the visual encoders, using a semantic encoder for understanding tasks while retaining a reconstruction-based tokenizer for generation, as demonstrated by Janus [79]. Show-O further refines this by integrating a discrete-diffusion schedule to improve token prediction [82]. More prevalent are hybrid architectures that combine the strengths of AR and diffusion models. One major category consists of Sequential AR-Diffusion models,

where an AR component generates an intermediate representation that conditions a diffusion-based decoder. In some variants, a pre-trained MLLM is kept frozen for reasoning, and its features are routed via learnable queries or hidden states to an external image generator [37,61,68]. This cascaded design effectively leverages powerful existing models. A more integrated approach uses a Unified Transformer Backbone [6, 92], where both AR and diffusion objectives are optimized simultaneously within a single transformer. To improve scalability, the Mixture-of-Transformers (MoT) paradigm has been introduced [11,35]. This approach, exemplified by Bagel, employs a sparse, modular design where specialized experts handle different modalities but share information through a common attention mechanism. Our work, SRUM, inherits this flexible MoT framework, demonstrating the versatility of our method on advanced UMMs.

##### 2.2 Post-Training Stage in UMMs

In addition to architectural innovations, considerable research has focused on the post-training stage to enhance the generative abilities of UMMs. Methods such as Chain-of-Thought (CoT) and test-time verification introduce explicit reasoning steps or iterative output validation [14,16,21]. However, these often depend on external models and do not fundamentally improve the native generative capacity of UMMs. Reinforcement learning techniques, including Direct Preference Optimization (DPO), leverage human or automated feedback to refine generation policies. While effective, these require carefully curated paired data and delicate advantage-function tuning with text-dependent rewards [20,52,54]. Reconstruction Alignment (RecA) introduces a post-training method based on reconstruction loss, yielding improved semantic understanding [81]. Some work has also attempted to use rule-level rewards for guidance, but such rewards are not universal and must be redesigned for different tasks [22,26,47]. In contrast, SRUM leverages the model’s inherent understanding to score self-generated samples, thereby enhancing performance without external reward models or human preference labels.

##### 2.3 Self-Rewarding in Understanding Models

Self-rewarding mechanisms have emerged as a significant paradigm for enhancing the understanding and reasoning capabilities of MLLMs. These approaches aim to reduce reliance on external preference data by enabling models to generate their own reward signals, thereby facilitating continuous self-improvement. For instance, CSR [93] achieves zero-cost self-enhancement through iterative online DPO with visual constraint rewards. SRPO [10] introduces a two-stage reflective reward mechanism, significantly improving the quality of reflection and answer accuracy in complex reasoning tasks. R1-Reward leverages process consistency rewards and stable reinforcement learning algorithms to enhance long-range reasoning stability [20]. Collectively, these works signal a paradigm shift from external rewards to self-criticism and optimization. However, they tend to focus

on a single dimension of feedback. Our SRUM framework proposes a more holistic approach. It distinguishes itself by incorporating a global-local dual reward system designed to provide a more comprehensive training signal.

### 3 SRUM: Self-Rewarding for Unified Multimodal Models

To drive self-improvement where the model’s understanding capabilities guide its generation abilities, we establish a multi-stage self-rewarding process. First, the Unified Multimodal Models (UMMs) generate high-quality candidate images with corresponding bounding boxes (as detailed in Section 3.1). Next, these candidates are evaluated by a global-local judgment framework that assesses both overall composition and fine-grained details, producing a holistic reward signal (Section 3.2). Finally, the cached rewards directly inform a reward-weighted training process, which enables targeted, region-specific optimization and reduces reward hacking (Section 3.3).

Step 1: Self-rewarding Data Generation Step 2: Reward-Weighted Training

Prompt: Traditional food of the Mid-Autumn Festival

- Moon Cake 1: Reward 0.95 Box (297, 569, 631, 884)
- Moon Cake 2: Reward -0.50 Box (697, 445, 957, 643) … Global Reward 𝜶: Reward 0.98

||0.20|
|---|
<br><br>0.95 0.90<br><br>-0.50|
|---|

|[Figure 9]<br><br>| | | |
|---|---|---|
| | | |
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
|
|---|

[Figure 10]

UMMs

###### UMMs

Understanding Generation

| |
|---|

| |
|---|

| |
|---|

[Figure 11]

− 2

| |
|---|

| |
|---|

UMMs

⨀

Generation

| |
|---|

| |
|---|

| |
|---|

GT Velocity

Predicted Velocity

Original Prompt: Traditional food of the Mid-Autumn Festival

- Fig. 2: Showcase of the SRUM pipeline. It consists of two main stages: Self-Rewarding Data Generation and Reward-Weighted Training. The first stage generates high-quality data and scores it to produce a reward for the next training stage for self-improvement.

##### 3.1 Image Candidates and Bounding Box Generation

We developed a self-data generation pipeline that enables our model to create its own training data, removing the need for external image resources. It starts with the UMM using its “think” mode (a form of CoT) to generate semantically rich images [11,75]. For Bagel, an external segmentation model proposes spatial supports for grounding [31]; for BLIP3o, native grounding can be used instead. In both cases, the UMM verifies boxes, matches prompt-relevant regions, and assigns all semantic rewards, so the external component is only a localization aid rather than an external reward model.

##### 3.2 Rewarding Process

Self-Judgment for Reliable Rewarding. A cornerstone of self-improvement is enabling the model’s internal understanding module to serve as a stable and reliable “evaluator”. To ensure the scores it generates are consistently trustworthy,

we designed a comprehensive self-judgment mechanism to assess image quality and prompt alignment [18, 39, 84, 91]. This dual-level judgment is key to guaranteeing a thorough assessment. First, a local judgment evaluates objectlevel fidelity and artifacts on a strict [−1.0,1.0] scoring scale. A mandatory “Reason” field elicits an interpretable rationale for the score, akin to chain-ofthought prompting [16,21], which further bolsters the reliability of the process. We enforce semantic grounding by verifying that identified objects correspond to prompt keywords, and a non-linear penalty maps severe distortions to a highpenalty negative range (e.g., -0.9 to -0.5) to better reflect human visual sensitivity. Subsequently, a global judgment evaluates the holistic composition and spatial alignment with the prompt’s intent. Crucially, for prompts lacking specific compositional directives (e.g., “a picture of a tree”), a neutral score range (e.g., -0.4 to 0.4) is applied, ensuring a fair assessment.

Reward Generation for Training. To serve as the core learning signal for self-improvement, the self-judgment scores are converted into a dense reward map. This step leverages the UMM’s grounding capabilities to produce finegrained local rewards for prompt-relevant regions and a single global reward for the whole image. The global score is normalized to [0,1] to avoid spurious positive signals from multiplying two negative values (Appendix Section C). During reward generation, the understanding module is frozen and receives no gradients. SRUM is therefore offline: it generates candidates, caches rewards, and then updates the trainable generation/flow parameters.

##### 3.3 Reward-Weighted Training

The reward-weighted training stage is where the model achieves self-improvement through training with rewards. The core objective is to translate the capabilities of the understanding module directly into the functionality of the generation module. By using fine-grained local rewards and layout-aware global rewards to weight the training objective, we guide the generator to learn more detailed and accurate patterns from the original data. This process is the key to bridging the gap between the model’s understanding and generation components, enabling the generator to benefit from the insights of the evaluator. The mechanism for this goal is a reward-weighted training objective, centered on the loss term Lr. This term operates on the model’s velocity prediction vθ, a standard practice in flow-based frameworks [40, 43]. The loss is modulated by two cached feedback signals from the frozen understanding module: a regional reward map R ∈ [−1,1] for localized refinement and a global scalar α for overall compositional quality. The product of these signals, α · R, weights the squared error between the predicted velocity vθ and the target velocity derived from the original latent xgt0 . This allows for fine-grained control, encouraging preservation where feedback is positive (α · R > 0) and promoting change where it is negative (α · R < 0):

Lr = E α · R ⊙ vθ − (ϵ − xgt0 ) 2 (1)

Second, to ensure the model’s output conforms to the desired overall structure and to prevent reward hacking, we introduce a reference constraint term, Lref.

This term acts as a regularizer, penalizing the squared ℓ2 distance to the target velocity of the artifact-free latent xgt0 :

Lref = E vθ − (ϵ − xgt0 ) 2 (2)

The final training objective is a weighted sum of these two losses, balanced by a hyperparameter λc. This is the complete objective used by SRUM; no hidden online reward loss or evaluator update is applied. The composite design enables targeted local refinement while maintaining global coherence:

LTotal = Lr + λc · Lref (3)

### 4 Analysis of Self-Rewarding

We validate our Self-Rewarding for Unified Multimodal Models (SRUM) method across various unified multimodal models (UMMs) and evaluation benchmarks. In particular, we investigate the following aspects:

- – Generality and Performance: SRUM achieves better performance on composition and delivers consistent gains across frameworks. (Table 1)
- – Component Efficacy: Ablation studies confirm each component of the SRUM framework makes a critical contribution to the overall performance. (Figure 3)
- – Generalization: SRUM demonstrates in-domain and out-of-domain generalization, indicating that improvements in generation can be transferred from understanding. (Table 5, Figure 6, Table 6)

##### 4.1 Experimental Setup

Model Architectures. We evaluate SRUM on two powerful open-source UMMs.

All experiments are conducted as a post-training phase, starting from the official pre-trained weights. Bagel [11] is a versatile UMM that serves as our primary model for comprehensive analysis, including main results, ablation studies, and generalization tests. We evaluate both its standard and CoT inference modes. BLIP3o [7,53,76] is another current SOTA UMM used to validate the generality and effectiveness of SRUM with frozen MLLM training. We focus on these two families because some AR-type models, such as Show-O and Janus, can exhibit biases stemming from under-leveraged generation or understanding capabilities.

Datasets and Benchmarks. Our experiments leverage several specialized datasets for training and evaluation to ensure a thorough and multi-faceted analysis. For consistent and objective scoring across all generation benchmarks, we employ QwenVL-2.5-32B/QwenVL-2.5-72B [3] as the designated multimodal evaluator. It is crucial to clarify that these external models serve strictly as objective evaluation metrics (akin to accuracy in classification or mIoU in segmentation) and are completely excluded from our core self-rewarding training pipeline. We also report traditional metrics following prior work [7]. Our experiment begins with instruction data sourced from the T2I-CompBench training

set [28]. For our primary evaluation, we use the standard split of the same benchmark to compare SRUM-enhanced models against leading T2I and UMM baselines. To assess generalization, we evaluate the model’s in-domain transferability on GenEval [18], which includes similar compositional challenges, and WISE [49], which evaluates knowledge-informed generation. Furthermore, we evaluate broader out-of-domain reasoning-informed capabilities on T2I-ReasonBench [65], a benchmark containing complex prompts that require knowledge beyond the training distribution.

##### 4.2 Main Results

Model 3D Spat. Color Compl. Nonspat. Num. Shape Spatial Text. Overall T2I Models

FLUX.1-dev 76.39 90.63 83.51 87.47 75.30 80.20 84.23 87.07 83.10 FLUX.1-schnell 79.38 84.53 81.96 85.55 72.82 82.20 85.49 86.38 82.29 SD-3-medium 77.83 91.63 84.73 86.12 72.80 83.72 88.20 89.03 84.26 SD-xl-base-1 72.25 77.75 75.00 85.28 57.14 72.18 77.08 78.38 74.38

Unified Multimodal Models

Janus-Pro 76.17 84.25 80.28 80.47 56.43 65.14 79.67 69.67 74.01 Show-O2 88.61 87.73 87.88 85.91 69.74 73.99 86.60 82.17 82.83 OmniGen2 82.21 92.22 86.87 88.51 72.00 83.95 90.07 90.88 85.84

BLIP3o 81.73 89.92 85.55 84.78 71.67 83.75 92.47 87.45 84.66 +SRUM 83.78 90.22 86.57 85.10 74.52 85.44 93.88 86.52 85.75

Bagel 77.98 89.30 83.32 85.03 70.40 81.94 81.52 87.93 82.18 +SRUM 83.10 92.90 88.69 88.47 78.52 84.23 86.92 89.57 86.55

Bagel(CoT) 84.66 88.85 86.10 85.64 75.36 84.33 82.71 88.07 84.46 +SRUM 88.60 92.90 91.31 90.48 80.12 84.47 89.93 89.15 88.37

- Table 1: Comprehensive T2I-CompBench Results. This table includes T2I and UMMs models. Models incorporating the SRUM are denoted with +SRUM. Bold indicate the highest score in each column. Green values indicate the improvements.

As shown in Table 1, our proposed method, SRUM, achieves consistent and substantial performance gains across various compositional generation tasks. Specifically, when evaluated in CoT mode, Bagel+SRUM attains an overall score of 88.37, ranking first among current UMM baselines. This marks a significant improvement of +3.91 points over the baseline Bagel with CoT, demonstrating the efficacy of our approach. The advantages of SRUM are particularly pronounced in categories demanding spatial and complex reasoning as well as numeracy. For instance, our method sets new SOTA scores in Spatial (93.88),

###### 3D Spatial (88.60) and Complex (91.31) reasoning, including 3D and actionbased prompts. Although we observe a slight drop in texture and color categories in some cases, the overall trend remains positive, likely because our algorithm

Attribute Binding (↑, %) Object Relationship (↑, %)

Model

Complex (↑, %) Color Shape Texture Spatial Non-Spat.

Unified Multimodal Models

Show-o 56.00 41.00 46.00 20.00 30.00 29.00 Janus-Pro-7B 63.59 35.28 49.36 20.61 30.85 35.59

Bagel 70.62 73.55 74.21 72.15 58.20 43.10

+SRUM 72.88 76.86 77.48 74.40 60.91 44.41 Bagel(CoT) 80.95 83.92 84.35 71.80 57.55 43.65

###### +SRUM 82.14 85.17 89.47 74.03 59.96 44.51

- Table 2: Detailed T2I-CompBench Evaluation via Traditional Metrics. This table compares various UMMs models using specialized traditional evaluators. Models incorporating the SRUM are denoted with +SRUM. Bold indicates the highest score in each column. All values are reported in percentage (%) (higher is better ↑).

does not over-optimize low-level information for certain objects. Additionally, we report traditional metrics in Table 2 to provide a multi-angle comparison. While these evaluators confirm the consistent gains of SRUM, the improvements appear less pronounced than those in the LLM-based evaluation (Table 1). This discrepancy stems from the inherent limitations of traditional models (e.g., CLIP or BLIP), which often suffer from “bag-of-words” effects and lack fine-grained logical parsing. In contrast, our Qwen-based evaluation offers a more solid and nuanced assessment by leveraging superior multimodal reasoning, capturing complex spatial and attribute relationships that traditional metrics tend to overlook.

##### 4.3 Empirical Study

We primarily employ three Bagel variants for analysis: Base Model, where Bagel’s open-source weights are used directly for inference; SFT Model, where Bagel generates images from training instructions and is then trained on this self-generated data; and SRUM Model, where the same self-generated data is trained with SRUM’s reward-weighted objective.

Computational Efficiency. A key advantage of SRUM is its practical efficiency. By parallelizing the understanding module, scoring a batch of 6K candidate images requires fewer than 4 GPU-hours on an NVIDIA H100. Furthermore, the loss computation introduces negligible overhead compared to standard SFT (12.5 GPU-hours for SRUM vs. 12.4 GPU-hours for SFT). Crucially, this selfrewarding mechanism eliminates the need for laborious manual data cleaning, making it highly scalable and practical given the current scarcity of high-quality image-text data.

Ablation Results. To further verify the effectiveness of our proposed reward configuration, we perform an ablation study on Bagel results on T2I-CompBench by systematically modifying the reward scheme. As shown in Figure 3 (Left), our full SRUM model achieves the highest overall accuracy, with the ablation results confirming the critical role of each component. Specifically, omitting the local

reward leads to a performance drop of 0.76 in CoT mode and 1.04 in standard inference mode, confirming the necessity of fine-grained spatial feedback that bounding boxes provide. Omitting the global reward leads to a notable decrease in performance, underscoring its importance for capturing the overarching coherence and compositional structure of the generated images. Removing the reference constraint also results in a significant drop, proving its value in ensuring training stability. This aligns with conclusions from post-training methods like DPO [54], where a constraint is essential to prevent significant policy deviation due to reward hacking. Furthermore, using a simple sparse reward leads to substantial performance degradation, reinforcing the necessity of a continuous, dense reward signal for richer gradient information. This is particularly evident as sparse reward schemes, such as Dance-GRPO [85], are ill-suited for providing granular regional feedback, which highlights the value of our dense reward design. Overall, this ablation confirms that the efficacy of our framework stems from the synergistic contributions of each component.

[Figure 12]

[Figure 13]

- Fig. 3: Left: Module Evaluation. We report the accuracy drop (∆ Acc. %) from our SRUM. Specifically, 0-1 Reward represents the sparse reward. Right: Hyperparameter evaluation on T2I-CompBench. We report the accuracy under different λc values in two modes: CoT and non-CoT.

In Figure 3 (Right), we analyze the effect of different reference-constraint ratios on the experimental outcomes. Across both Bagel with CoT and nonCoT configurations, the results consistently indicate that λc = 0.5 is the most effective choice. Consequently, we keep this hyperparameter fixed in subsequent experiments.

Method UR ReCA Qwen Qwen Qwen Qwen SRUM (Ours) Size 7B - 7B 7B 32B 32B 7B

Config. - - w/o GR w/ GR w/o GR w/ GR Self-Rewarding Overall Score 85.15 86.45 85.78 87.52 85.53 88.01 88.37

- Table 3: Comparison with external reward models and post-training methods on T2ICompBench. UR and GR denote UnifiedReward and Global Reward, respectively. Our SRUM achieves the highest overall score.

Comparison with External Rewards and Post-Training Methods. To further validate the necessity and superiority of the self-rewarding mechanism, we compare SRUM against popular post-training baselines including Re-

construction Alignment (ReCA) and the RL-based UnifiedReward, as well as external VLM reward models (QwenVL2.5-7B and 32B). As shown in Table 3, SRUM significantly outperforms ReCA and UnifiedReward. Counter-intuitively, replacing our internal understanding module with powerful external VLMs does not yield stable improvements. On saved image/prompt pairs, the internal scorer achieves stronger calibration with the final evaluator (ρ = .58 vs. .43) and better accept/reject separation (0.67 vs. -0.31), explaining why it can be a better training signal even when external VLMs are larger. This shows that SRUM’s effectiveness comes from the intrinsic multi-scale design rather than external parameter scale.

Further Analysis. For a more granular investigation, we leverage the same powerful MLLM, QwenVL-2.5-72B, from our primary evaluation to conduct a deeper analysis of our method and the baseline. Specifically, we employ the MLLM to perform a step-by-step scoring of the inference process.

[Figure 14]

- (a) Detail score per step during inference

[Figure 15]

- (b) Layout score per step during inference

Fig. 4: Score per step during inference in Bagel with its ablation models.

The evaluation is divided into two metrics: (1) layout, which assesses the concordance of the overall structure and quality, and (2) detail, which measures the fidelity of the generated fine-grained details. Our ablation study, visualized in Figure 4, systematically isolates the effects of each component. We observe that the “think” mode primarily bolsters the initial layout generation by improving the high-level reasoning process. The global reward component of SRUM then further refines this layout during the early stages of inference. In contrast, a baseline using only this global reward (labeled “sample reward”) yields negligible improvements in detail fidelity. This highlights a crucial finding: the finegrained, local rewards are essential for the subsequent optimization of details, with their benefits becoming most apparent in the later inference steps. Collectively, these results demonstrate that our dual global-local reward mechanism provides a multi-stage optimization path: first establishing a coherent layout and

then progressively refining the details. This synergistic approach allows SRUM to significantly outperform standard SFT on the same self-generated data.

Impact on Understanding Module. As shown in Table 4, our method has a minimal impact on the model’s core understanding capabilities. On prevalent benchmarks such as MME [17], MM-Vet [87], MMBench [44], MMMU [88], and MathVista [46], the results exhibit only marginal fluctuations compared to the base version. Notably, performance on MMVP [69] even improves, which is consistent with prior works [68, 71, 73]. This indicates that our method holds significant potential for further iterative enhancement. In Figure 5, we track the activation dynamics of two distinct functional clusters, Understanding and Generation, across the Base, SFT, and SRUM models. In Bagel’s inference, the parameters primarily activated during understanding tasks are defined as the understanding cluster, and those primarily activated during generation tasks are defined as the generation cluster. Our analysis reveals two distinct finetuning paradigms. Conventional SFT exhibits a narrowing effect, achieving specialization by suppressing irrelevant functional clusters. In contrast, SRUM demonstrates an enhancing and orchestrating effect, strengthening the primary taskrelevant cluster while maintaining supportive activation in secondary clusters. This promotes robust and generalizable representations. Detailed settings are provided in Appendix B.

[Figure 16]

###### Base SFT SRUM

MME-P 1687 1682 1673 MME-C 701 683 677 MMBench 85.0 84.6 84.8 MM-Vet 67.2 66.5 67.0 MMMU 55.3 55.0 55.2 MathVista 73.1 72.8 73.0 MMVP 69.3 68.7 70.0

- Fig. 5: Functional cluster activation patterns of the different models (Bagel, SFT and SRUM) on different understanding and generation tasks. The average activation strength of Understanding and Generation clusters is shown.

Table 4: Comparison with the results of different models (Bagel, SFT and SRUM) on understanding benchmarks. MME-P and MME-C represent the perception and cognition parts, respectively.

In-Domain Generalization. We then investigate the in-domain generalization capability of our model. We posit that the compositional abilities learned from the T2I-CompBench training set should be transferable to other benchmarks with similar evaluation perspectives. To test this hypothesis, we evaluate SRUM on the GenEval benchmark without any further fine-tuning. The comparative results are summarized in Table 5.

As shown in the table, our evaluation on GenEval further validates the strengths of SRUM, particularly in the challenging domain of object counting. SRUM attains the highest score of 0.83 in Counting, surpassing both the base model and the SFT baseline. Crucially, this superior performance in numeri-

cal generation aligns with our previous results on T2I-CompBench. This consistency across benchmarks underscores our method’s reliable improvement in processing quantitative information. By excelling at a complex task like counting while retaining proficiency in simpler ones, the model demonstrates strong indomain generalization. This confirms that the targeted enhancements by SRUM are transferable.

Model Single obj. Two obj. Counting Colors Position Color attr.

Bagel 0.99 0.94 0.81 0.88 0.64 0.82 Bagel+SFT 0.96 0.94 0.79 0.92 0.59 0.78 Bagel+SRUM 0.98 0.94 0.83 0.90 0.64 0.83

- Table 5: Results on key visual attributes at GenEval. For brevity, some model names are shortened; abbreviations are defined at the beginning of Section 4.3. Bold values are the best in each column.

In-Domain Knowledge-based Generalization. Following this, we explore whether our method holds a distinct advantage for reasoning-based generation, a current area of focus in the community. We train the model on one category of prompts from the WISE benchmark and perform in-domain evaluations on the remaining two categories. This protocol yields three distinct evaluation sets for analyzing the model’s generalization capabilities.

As illustrated in Figure 6, selecting any single group for training generally enhances the image generation performance of the other two groups. This improvement is consistent across both standard and CoT reasoning paradigms. It shows that SRUM can promote knowledge-domain generalization, enabling generation to better fit instruction semantics at the knowledge level.

[Figure 17]

- Fig. 6: Results of Bagel on WISE. We train on one of the three WISE tasks and evaluate on the other two, showing SRUM’s knowledge-domain generalization.

Out-of-Domain Generalization. To further evaluate the generalization capability of our model on unseen domains, we use T2I-ReasonBench, a large-scale benchmark for analyzing the reasoning quality of generated images. In this experiment, we take the model trained with T2I-CompBench prompts and directly evaluate its performance on this benchmark. This setup demonstrates the model’s ability to generalize to advanced, reasoning-informed image generation tasks. We primarily focus on accuracy scores, which measure the

model’s high-level semantic alignment with the given prompts. To prevent selfrewrite from directly parsing hidden high-level semantics (e.g., in the Idiom category, a phrase like “a piece of cake” might be literally interpreted as “easy,” which would obscure the model’s ability to transfer understanding and could interfere with evaluation), we use Bagel without CoT during evaluation.

As illustrated in Table 6, SRUM achieves superior prompt understanding compared with both the SFT and Base models. While SFT also yields a noticeable improvement, the enhanced performance of SRUM demonstrates that our approach improves generalization on complex problems from both a data and an algorithmic perspective. Furthermore, for image-based prompts, SRUM provides consistent improvements, in contrast to the volatility exhibited by the SFT model. This further substantiates that our algorithmic design is more adaptable and accounts for more nuanced factors. Finally, visual case studies and a detailed analysis of failure modes are provided in Appendix D, helping differentiate algorithmic boundaries from inherent generative model limitations.

Model Entity Idiom Scientific Textual Overall

Bagel 49.70 34.46 47.52 43.59 43.82 Bagel+SFT 50.53 39.43 47.45 44.08 45.37 Bagel+SRUM 52.85 40.51 47.83 45.83 46.75

- Table 6: Performance comparison of Bagel models across four categories and their overall scores. Bold values indicate the best performance in each column. Scores are normalized between 0-100.

### 5 Conclusion

This paper introduces SRUM, a fine-grained post-training framework that enables a model’s understanding module to reward its generation module. Additionally, SRUM decomposes the reward into local and global components, facilitating multi-scale alignment and refinement. Extensive experiments validate SRUM’s effectiveness, setting new state-of-the-art results on complex compositional and reasoning benchmarks such as T2I-CompBench and T2I-ReasonBench. The framework demonstrates robust in-domain and out-of-domain generalization, and our empirical analysis confirms the efficacy of the fine-grained reward design. These findings illuminate the synergistic development of understanding and generation capabilities within a single model and establish the principle of self-reward as a promising direction for future research.

SRUM is a preliminary exploration of self-rewarding for Unified Multimodal Models (UMMs). There remains room to improve the prompts used by the understanding module during scoring, and we plan to scale the method to larger datasets. Although the current implementation uses standardized prompt templates and, for some models, external localization aids, the reward signal itself comes from the UMM’s frozen understanding module. A natural next step is to let the understanding module self-play questions and answers to build a more closed-loop training system.

### Acknowledgements

This work is supported by the Shenzhen Loop Area Institute under grant FPF 10120250006, and partially supporteed under grants FPFFPF10120250002.

### References

- 1. Ainslie, J., Lee-Thorp, J., De Jong, M., Zemlyanskiy, Y., Lebrón, F., Sanghai, S.: Gqa: Training generalized multi-query transformer models from multi-head checkpoints. arXiv preprint arXiv:2305.13245 (2023)
- 2. Bai, J., Chow, W., Yang, L., Li, X., Li, J., Zhang, H., Yan, S.: Humanedit: A high-quality human-rewarded dataset for instruction-based image editing. arXiv preprint arXiv:2412.04280 (2024)
- 3. Bai, S., Chen, K., Liu, X., Wang, J., Ge, W., Song, S., Dang, K., Wang, P., Wang, S., Tang, J., Zhong, H., Zhu, Y., Yang, M., Li, Z., Wan, J., Wang, P., Ding, W., Fu, Z., Xu, Y., Ye, J., Zhang, X., Xie, T., Cheng, Z., Zhang, H., Yang, Z., Xu, H., Lin, J.: Qwen2.5-vl technical report. arXiv preprint arXiv:2502.13923 (2025)
- 4. Betker, J., Goh, G., Jing, L., Brooks, T., Wang, J., Li, L., Ouyang, L., Zhuang, J., Lee, J., Guo, Y., et al.: Improving image generation with better captions. OpenAI blog (2023)
- 5. Brooks, T., Holynski, A., Efros, A.A.: Instructpix2pix: Learning to follow image editing instructions. In: CVPR (2023)
- 6. Chen, B., Martí Monsó, D., Du, Y., Simchowitz, M., Tedrake, R., Sitzmann, V.: Diffusion forcing: Next-token prediction meets full-sequence diffusion. In: NeurIPS

(2024)

- 7. Chen, J., Xu, Z., Pan, X., Hu, Y., Qin, C., Goldstein, T., Huang, L., Zhou, T., Xie, S., Savarese, S., et al.: Blip3-o: A family of fully open unified multimodal models-architecture, training and dataset. arXiv preprint arXiv:2505.09568 (2025)
- 8. Chen, X., Wu, C., Wu, Z., Ma, Y., Liu, X., Pan, Z., Liu, W., Xie, Z., Yu, X., Ruan, C., Luo, P.: Janus-pro: Unified multimodal understanding and generation with data and model scaling. arXiv preprint arXiv:2501.17811 (2025)
- 9. Chen, Z., Wang, W., Tian, H., Ye, S., Gao, Z., Cui, E., Tong, W., Hu, K., Luo, J., Ma, Z., et al.: How far are we to gpt-4v? closing the gap to commercial multimodal models with open-source suites. SCIS (2024)
- 10. Choi, E., Ahmadian, A., Geist, M., Pietquin, O., Azar, M.G.: Self-improving robust preference optimization. arXiv preprint arXiv:2406.01660 (2024)
- 11. Deng, C., Zhu, D., Li, K., Gou, C., Li, F., Wang, Z., Zhong, S., Yu, W., Nie, X., Song, Z., et al.: Emerging properties in unified multimodal pretraining. arXiv preprint arXiv:2505.14683 (2025)
- 12. Dong, R., Han, C., Peng, Y., Qi, Z., Ge, Z., Yang, J., Zhao, L., Sun, J., Zhou, H., Wei, H., et al.: Dreamllm: Synergistic multimodal comprehension and creation. In: ICLR (2024)
- 13. Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J., Houlsby, N.: An image is worth 16x16 words: Transformers for image recognition at scale. In: ICLR (2021)
- 14. Duan, C., Fang, R., Wang, Y., Wang, K., Huang, L., Zeng, X., Li, H., Liu, X.: Got-r1: Unleashing reasoning capability of mllm for visual generation with reinforcement learning. arXiv preprint arXiv:2505.17022 (2025)

- 15. Esser, P., Kulal, S., Blattmann, A., Entezari, R., Müller, J., Saini, H., Levi, Y., Lorenz, D., Sauer, A., Boesel, F., et al.: Scaling rectified flow transformers for high-resolution image synthesis. In: ICML (2024)
- 16. Fang, R., Duan, C., Wang, K., Huang, L., Li, H., Yan, S., Tian, H., Zeng, X., Zhao, R., Dai, J., et al.: Got: Unleashing reasoning capability of multimodal large language model for visual generation and editing. arXiv preprint arXiv:2503.10639

(2025)

- 17. Fu, C., Chen, P., Shen, Y., Qin, Y., Zhang, M., Lin, X., Yang, J., Zheng, X., Li, K., Sun, X., et al.: Mme: A comprehensive evaluation benchmark for multimodal large language models. arXiv preprint arXiv:2306.13394 (2023)
- 18. Ghosh, D., Hajishirzi, H., Schmidt, L.: Geneval: An object-focused framework for evaluating text-to-image alignment. In: NeurIPS (2023)
- 19. Goyal, P.: Accurate, large minibatch sg d: training imagenet in 1 hour. arXiv preprint arXiv:1706.02677 (2017)
- 20. Guo, D., Yang, D., Zhang, H., Song, J., Zhang, R., Xu, R., Zhu, Q., Ma, S., Wang, P., Bi, X., et al.: Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948 (2025)
- 21. Guo, Z., Zhang, R., Tong, C., Zhao, Z., Huang, R., Zhang, H., Zhang, M., Liu, J., Zhang, S., Gao, P., et al.: Can we generate images with cot? let’s verify and reinforce image generation step by step. arXiv preprint arXiv:2501.13926 (2025)
- 22. Han, Y., Chen, H., Han, A., Wang, Z., Lin, X., Zhang, Y., Zhang, S., Zou, D.: Selfcontradiction as self-improvement: Mitigating the generation-understanding gap in mllms. arXiv preprint arXiv:2507.16663 (2025)
- 23. Ho, J., Jain, A., Abbeel, P.: Denoising diffusion probabilistic models. In: NeurIPS

(2020)

- 24. Ho, J., Salimans, T.: Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598 (2022)
- 25. Hoffmann, J., Borgeaud, S., Mensch, A., Buchatskaya, E., Cai, T., Rutherford, E., de Las Casas, D., Hendricks, L.A., Welbl, J., Clark, A., Hennigan, T., Noland, E., Millican, K., van den Driessche, G., Damoc, B., Guy, A., Osindero, S., Simonyan, K., Elsen, E., Rae, J.W., Vinyals, O., Sifre, L.: Training compute-optimal large language models. arXiv preprint arxiv:2203.15556 (2022)
- 26. Hong, J., Zhang, Y., Wang, G., Liu, Y., Wen, J.R., Yan, R.: Reinforcing multimodal understanding and generation with dual self-rewards. arXiv preprint arXiv:2506.07963 (2025)
- 27. Hu, S., Tu, Y., Han, X., Cui, G., He, C., Zhao, W., Long, X., Zheng, Z., Fang, Y., Huang, Y., Zhang, X., Thai, Z.L., Wang, C., Yao, Y., Zhao, C., Zhou, J., Cai, J., Zhai, Z., Ding, N., Jia, C., Zeng, G., dahai li, Liu, Z., Sun, M.: Minicpm: Unveiling the potential of small language models with scalable training strategies. In: COLM

(2024)

- 28. Huang, K., Sun, K., Xie, E., Li, Z., Liu, X.: T2i-compbench: A comprehensive benchmark for open-world compositional text-to-image generation. Advances in Neural Information Processing Systems 36, 78723–78747 (2023)
- 29. Hui, M., Yang, S., Zhao, B., Shi, Y., Wang, H., Wang, P., Zhou, Y., Xie, C.: Hqedit: A high-quality dataset for instruction-based image editing. arXiv preprint arXiv:2404.09990 (2024)
- 30. Kaplan, J., McCandlish, S., Henighan, T., Brown, T., Chess, B., Child, R., Gray, S., Radford, A., Wu, J., Amodei, D.: Scaling laws for neural language models. In: ICML (2020)

- 31. Kirillov, A., Mintun, E., Ravi, N., Mao, H., Rolland, C., Gustafson, L., Xiao, T., Whitehead, S., Berg, A.C., Lo, W.Y., et al.: Segment anything. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 4015–4026 (2023)
- 32. Labs, B.F.: Flux (2024), https://github.com/black-forest-labs/flux
- 33. Li, Q., Chen, Z., Wang, W., Wang, W., Ye, S., Jin, Z., Chen, G., He, Y., Gao, Z., Cui, E., et al.: Omnicorpus: A unified multimodal corpus of 10 billion-level images interleaved with text. arXiv preprint arXiv:2406.08418 (2024)
- 34. Li, X., Tu, H., Hui, M., Wang, Z., Zhao, B., Xiao, J., Ren, S., Mei, J., Liu, Q., Zheng, H., et al.: What if we recaption billions of web images with llama-3? arXiv preprint arXiv:2406.08478 (2024)
- 35. Liang, W., YU, L., Luo, L., Iyer, S., Dong, N., Zhou, C., Ghosh, G., Lewis, M., tau Yih, W., Zettlemoyer, L., Lin, X.V.: Mixture-of-transformers: A sparse and scalable architecture for multi-modal foundation models. TMLR (2025)
- 36. Liao, J., Niu, Y., Meng, F., Li, H., Tian, C., Du, Y., Xiong, Y., Li, D., Zhu, X., Yuan, L., et al.: Langbridge: Interpreting image as a combination of language embeddings. arXiv preprint arXiv:2503.19404 (2025)
- 37. Lin, B., Li, Z., Cheng, X., Niu, Y., Ye, Y., He, X., Yuan, S., Yu, W., Wang, S., Ge, Y., et al.: Uniworld: High-resolution semantic encoders for unified visual understanding and generation. arXiv preprint arXiv:2506.03147 (2025)
- 38. Lin, T.Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D., Dollár, P., Zitnick, C.L.: Microsoft coco: Common objects in context. In: ECCV (2014)
- 39. Lin, Z., Pathak, D., Li, B., Li, J., Xia, X., Neubig, G., Zhang, P., Ramanan, D.: Evaluating text-to-visual generation with image-to-text generation. In: European Conference on Computer Vision. pp. 366–384. Springer (2024)
- 40. Lipman, Y., Chen, R.T., Ben-Hamu, H., Nickel, M., Le, M.: Flow matching for generative modeling. In: ICLR (2023)
- 41. Liu, H., Li, C., Wu, Q., Lee, Y.J.: Visual instruction tuning. NeurIPS 36, 34892– 34916 (2023)
- 42. Liu, S., Cheng, H., Liu, H., Zhang, H., Li, F., Ren, T., Zou, X., Yang, J., Su, H., Zhu, J., et al.: Llava-plus: Learning to use tools for creating multimodal agents. In: ECCV (2024)
- 43. Liu, X., Gong, C., et al.: Flow straight and fast: Learning to generate and transfer data with rectified flow. In: ICLR (2023)
- 44. Liu, Y., Duan, H., Zhang, Y., Li, B., Zhang, S., Zhao, W., Yuan, Y., Wang, J., He, C., Liu, Z., et al.: Mmbench: Is your multi-modal model an all-around player? In: ECCV (2024)
- 45. Loshchilov, I.: Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101 (2017)
- 46. Lu, P., Bansal, H., Xia, T., Liu, J., Li, C., Hajishirzi, H., Cheng, H., Chang, K.W., Galley, M., Gao, J.: Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. In: NeurIPS Workshop on Mathematical Reasoning and AI (2023)
- 47. Mao, W., Yang, Z., Shou, M.Z.: Unirl: Self-improving unified multimodal models via supervised and reinforcement learning. arXiv preprint arXiv:2505.23380 (2025)
- 48. Molybog, I., Albert, P., Chen, M., DeVito, Z., Esiobu, D., Goyal, N., Koura, P.S., Narang, S., Poulton, A., Silva, R., et al.: A theory on adam instability in large-scale machine learning. arXiv preprint arXiv:2304.09871 (2023)
- 49. Niu, Y., Ning, M., Zheng, M., Lin, B., Jin, P., Liao, J., Ning, K., Zhu, B., Yuan, L.: Wise: A world knowledge-informed semantic evaluation for text-to-image generation. arXiv preprint arXiv:2503.07265 (2025)

- 50. Pan, X., Shukla, S.N., Singh, A., Zhao, Z., Mishra, S.K., Wang, J., Xu, Z., Chen, J., Li, K., Juefei-Xu, F., Hou, J., Xie, S.: Transfer between modalities with metaqueries. arXiv preprint arXiv:2504.06256 (2025)
- 51. Podell, D., English, Z., Lacey, K., Blattmann, A., Dockhorn, T., Müller, J., Penna, J., Rombach, R.: Sdxl: Improving latent diffusion models for high-resolution image synthesis. In: ICLR (2024)
- 52. Qu, L., Li, H., Wang, W., Liu, X., Li, J., Nie, L., Chua, T.S.: Silmm: Self-improving large multimodal models for compositional text-to-image generation. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 18497–18508

(2025)

- 53. Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al.: Learning transferable visual models from natural language supervision. In: International conference on machine learning. pp. 8748–8763. PmLR (2021)
- 54. Rafailov, R., Sharma, A., Mitchell, E., Manning, C.D., Ermon, S., Finn, C.: Direct preference optimization: Your language model is secretly a reward model. Advances in neural information processing systems 36, 53728–53741 (2023)
- 55. Ramesh, A., Pavlov, M., Goh, G., Gray, S., Voss, C., Radford, A., Chen, M., Sutskever, I.: Zero-shot text-to-image generation. In: ICML (2021)
- 56. Rombach, R., Blattmann, A., Lorenz, D., Esser, P., Ommer, B.: High-resolution image synthesis with latent diffusion models. In: CVPR. pp. 10684–10695 (2022)
- 57. Saharia, C., Chan, W., Saxena, S., Li, L., Whang, J., Denton, E.L., Ghasemipour, K., Gontijo Lopes, R., Karagol Ayan, B., Salimans, T., et al.: Photorealistic textto-image diffusion models with deep language understanding. In: NeurIPS (2022)
- 58. Schuhmann, C., Beaumont, R., Vencu, R., Gordon, C., Wightman, R., Cherti, M., Coombes, T., Katta, A., Mullis, C., Wortsman, M., et al.: Laion-5b: An open large-scale dataset for training next generation image-text models. In: NeurIPS

(2022)

- 59. Sharma, P., Ding, N., Goodman, S., Soricut, R.: Conceptual captions: A cleaned, hypernymed, image alt-text dataset for automatic image captioning. In: ACL

(2018)

- 60. Shazeer, N.: Glu variants improve transformer. arXiv preprint arXiv:2002.05202

(2020)

- 61. Shi, W., Han, X., Zhou, C., Liang, W., Lin, X.V., Zettlemoyer, L., Yu, L.: Llamafusion: Adapting pretrained language models for multimodal generation. arXiv preprint arXiv:2412.15188 (2024)
- 62. Sohl-Dickstein, J., Weiss, E., Maheswaranathan, N., Ganguli, S.: Deep unsupervised learning using nonequilibrium thermodynamics. In: ICML. pp. 2256–2265. PMLR (2015)
- 63. Song, Y., Sohl-Dickstein, J., Kingma, D.P., Kumar, A., Ermon, S., Poole, B.: Scorebased generative modeling through stochastic differential equations. In: ICLR

(2021)

- 64. Su, J., Ahmed, M., Lu, Y., Pan, S., Bo, W., Liu, Y.: Roformer: Enhanced transformer with rotary position embedding. Neurocomputing 568, 127063 (2024)
- 65. Sun, K., Fang, R., Duan, C., Liu, X., Liu, X.: T2i-reasonbench: Benchmarking reasoning-informed text-to-image generation. arXiv preprint arXiv:2508.17472

(2025)

- 66. Team, C.: Chameleon: Mixed-modal early-fusion foundation models. arXiv preprint arXiv:2405.09818 (2024)

- 67. Team, G., Anil, R., Borgeaud, S., Alayrac, J.B., Yu, J., Soricut, R., Schalkwyk, J., Dai, A.M., Hauth, A., Millican, K., et al.: Gemini: a family of highly capable multimodal models. arXiv preprint arxiv:2312.11805 (2023)
- 68. Tong, S., Fan, D., Zhu, J., Xiong, Y., Chen, X., Sinha, K., Rabbat, M., LeCun, Y., Xie, S., Liu, Z.: Metamorph: Multimodal understanding and generation via instruction tuning. arXiv preprint arXiv:2412.14164 (2024)
- 69. Tong, S., Liu, Z., Zhai, Y., Ma, Y., LeCun, Y., Xie, S.: Eyes wide shut? exploring the visual shortcomings of multimodal llms. In: CVPR. pp. 9568–9578 (2024)
- 70. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, L., Polosukhin, I.: Attention is all you need. In: NeurIPS (2017)
- 71. Wang, H., Zheng, A., Zhao, Y., Wang, T., Ge, Z., Zhang, X., Zhang, Z.: Reconstructive visual instruction tuning. arXiv preprint arXiv:2410.09575 (2024)
- 72. Wang, P., Bai, S., Tan, S., Wang, S., Fan, Z., Bai, J., Chen, K., Liu, X., Wang, J., Ge, W., Fan, Y., Dang, K., Du, M., Ren, X., Men, R., Liu, D., Zhou, C., Zhou, J., Lin, J.: Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191 (2024)
- 73. Wang, W., Sun, Q., Zhang, F., Tang, Y., Liu, J., Wang, X.: Diffusion feedback helps clip see better. arXiv preprint arXiv:2407.20171 (2024)
- 74. Wang, X., Zhang, X., Luo, Z., Sun, Q., Cui, Y., Wang, J., Zhang, F., Wang, Y., Li, Z., Yu, Q., et al.: Emu3: Next-token prediction is all you need. arXiv preprint arxiv:2409.18869 (2024)
- 75. Wang, Y., Liu, M., He, W., Zhang, L., Huang, Z., Zhang, G., Shu, F., Tao, Z., She, D., Yu, Z., et al.: Mint: Multi-modal chain of thought in unified generative models for enhanced image generation. arXiv preprint arXiv:2503.01298 (2025)
- 76. Wang, Z., Li, Y., Chen, X., Lim, S.N., Torralba, A., Zhao, H., Wang, S.: Detecting everything in the open world: Towards universal object detection (2023)
- 77. Wei, C., Xiong, Z., Ren, W., Du, X., Zhang, G., Chen, W.: Omniedit: Building image editing generalist models through specialist supervision. In: ICLR (2024)
- 78. Wu, C., Li, J., Zhou, J., Lin, J., Gao, K., Yan, K., Yin, S.m., Bai, S., Xu, X., Chen, Y., et al.: Qwen-image technical report. arXiv preprint arXiv:2508.02324 (2025)
- 79. Wu, C., Chen, X., Wu, Z., Ma, Y., Liu, X., Pan, Z., Liu, W., Xie, Z., Yu, X., Ruan, C., Luo, P.: Janus: Decoupling visual encoding for unified multimodal understanding and generation. arXiv preprint arXiv:2410.13848 (2024)
- 80. Wu, Y., Zhang, Z., Chen, J., Tang, H., Li, D., Fang, Y., Zhu, L., Xie, E., Yin, H., Yi, L., Han, S., Lu, Y.: Vila-u: A unified foundation model integrating visual understanding and generation. arXiv preprint arXiv:2409.04429 (2024), https: //arxiv.org/abs/2409.04429
- 81. Xie, J., Darrell, T., Zettlemoyer, L., Wang, X.: Reconstruction alignment improves unified multimodal models (2025), https://arxiv.org/abs/2509.07295
- 82. Xie, J., Mao, W., Bai, Z., Zhang, D.J., Wang, W., Lin, K.Q., Gu, Y., Chen, Z., Yang, Z., Shou, M.Z.: Show-o: One single transformer to unify multimodal understanding and generation. arXiv preprint arxiv:2408.12528 (2024)
- 83. Xie, J., Yang, Z., Shou, M.Z.: Show-o2: Improved native unified multimodal models. arXiv preprint arXiv:2506.15564 (2025)
- 84. Xu, J., Liu, X., Wu, Y., Tong, Y., Li, Q., Ding, M., Tang, J., Dong, Y.: Imagereward: Learning and evaluating human preferences for text-to-image generation. Advances in Neural Information Processing Systems 36, 15903–15935 (2023)
- 85. Xue, Z., Wu, J., Gao, Y., Kong, F., Zhu, L., Chen, M., Liu, Z., Liu, W., Guo, Q., Huang, W., et al.: Dancegrpo: Unleashing grpo on visual generation. arXiv preprint arXiv:2505.07818 (2025)

- 86. Yu, Q., Chow, W., Yue, Z., Pan, K., Wu, Y., Wan, X., Li, J., Tang, S., Zhang, H., Zhuang, Y.: Anyedit: Mastering unified high-quality image editing for any idea. arXiv preprint arXiv:2411.15738 (2024)
- 87. Yu, W., Yang, Z., Li, L., Wang, J., Lin, K., Liu, Z., Wang, X., Wang, L.: Mm-vet: Evaluating large multimodal models for integrated capabilities. In: ICML (2024)
- 88. Yue, X., Ni, Y., Zhang, K., Zheng, T., Liu, R., Zhang, G., Stevens, S., Jiang, D., Ren, W., Sun, Y., et al.: Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. In: CVPR (2024)
- 89. Zhang, B., Sennrich, R.: Root mean square layer normalization. In: NeurIPS (2019)
- 90. Zhang, K., Mo, L., Chen, W., Sun, H., Su, Y.: Magicbrush: A manually annotated dataset for instruction-guided image editing. In: NeurIPS (2023)
- 91. Zhang, W., Zhai, G., Wei, Y., Yang, X., Ma, K.: Blind image quality assessment via vision-language correspondence: A multitask learning perspective. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 14071–14081 (2023)
- 92. Zhao, C., Song, Y., Wang, W., Feng, H., Ding, E., Sun, Y., Xiao, X., Wang, J.: Monoformer: One transformer for both diffusion and autoregression. arXiv preprint arXiv:2409.16280 (2024), https://arxiv.org/abs/2409.16280
- 93. Zhou, Y., Fan, Z., Cheng, D., Yang, S., Chen, Z., Cui, C., Wang, X., Li, Y., Zhang, L., Yao, H.: Calibrated self-rewarding vision language models. Advances in Neural Information Processing Systems 37, 51503–51531 (2024)

Hyperparameters Post-training Learning rate 2.5 × 10−5 LR scheduler Constant Weight decay 0.0 Gradient norm clip 1.0 Optimizer AdamW (β1 = 0.9, β2 = 0.95, ϵ = 10−15) Warm-up steps 500 Max context window 40k Gen resolution (min short side, max long side) (512, 1024) Diffusion timestep shift 4.0

- Table 7: Training recipe of SRUM. We report the specific hyperparameter settings used during the post-training stage.

### A Detail Settings

Following the configuration of stage 4 from the Bagel [11] framework during our post-training phase, we employed the AdamW optimizer [45], configured with momentum parameters β1 = 0.9 and β2 = 0.95. Drawing inspiration from [48], we set the epsilon value to 1.0×10−15 to mitigate loss spikes. When we increase the resolution during generation, we also adjust the diffusion timestep from 1.0 to 4.0, which helps maintain a stable noise-level distribution. We chose a constant learning rate, as this approach, as suggested by [27], simplifies the scaling of training data without needing to restart the training process. These empirical observations, along with established practices for large-scale model training [19, 25,30,36], informed our final training protocol.

Our model architecture builds upon the standard Transformer [70] and ViT [13]

paradigms, incorporating modern enhancements for stability and efficiency, such as RMS Layer Normalization [89], GLU variants for activation functions [60], RoPE [64], and GQA [1]. The generative process is fundamentally based on principles from diffusion process [23,62,63], and utilizes classifier-free guidance [24] within a latent space [56] for high-resolution synthesis. The complete training recipe is summarized in Table 7.

In Section 3.1, we explain how detection boxes are generated. Bagel uses an external localization aid (SAM), while BLIP3o can rely on its native grounding capabilities. In both cases, the UMM verifies the boxes and supplies the semantic reward, so the external component is not an external reward model. The choice of localization source can be guided by the model’s performance on grounding benchmarks such as RefCOCO.

### B Definition and Calculation of Average Activation Strength

To investigate the internal functional mechanisms of different training methods, we introduce the metric of Average Activation Strength. This metric is designed

to quantify the overall activity level of a predefined functional neural cluster when the model is performing a specific type of task. This appendix provides a detailed definition, mathematical formulation, and the statistical implementation procedure. The Average Activation Strength is defined as the mean activation value of all neurons within a specific functional cluster, averaged over an entire dataset for a given task. The calculation involves a two-level averaging process:

- 1. Intra-Cluster Average: For a single input sample, we compute the mean of the activation values of all neurons belonging to the target cluster.
- 2. Dataset-Wide Average: We then average these single-sample cluster means across all samples in the entire task dataset.

This metric reflects the degree of engagement of a functional cluster (e.g., the “Understanding Cluster”) while processing a certain category of tasks (e.g., “Generation Tasks”). A higher value indicates that the cluster is more strongly and broadly activated for that task.

To formalize this definition, we first introduce the following notation:

- – M: A specific neural network model (e.g., Base, SFT, or SRUM).
- – Ck: A functional neural cluster k (e.g., Cunderstand or Cgenerate), which is a set of specific neuron indices.
- – |Ck|: The number of neurons in cluster Ck.
- – DT: The dataset for a specific task type T (e.g., Dunderstanding or Dgeneration).
- – |DT|: The number of samples in the dataset DT.
- – x: An individual input sample from the dataset, where x ∈ DT.
- – ai(x): The activation value of neuron i in model M given the input x, where i ∈ Ck. This typically refers to the output of a neuron after its activation function (e.g., ReLU or GeLU) has been applied.

For a single input sample x, the average activation strength of a cluster Ck, denoted as Ssample, is calculated as:

1 |Ck| i∈C

Ssample(M,Ck,x) =

k

ai(x) (4)

The final Average Activation Strength of cluster Ck for model M over the entire dataset DT, denoted as Sfinal, is the expected value of Ssample over all samples. In practice, this is estimated by averaging across the dataset:

1 |DT| x∈D

1 |DT||Ck| x∈D

Sfinal(M,Ck,DT) =

Ssample(M,Ck,x) =

ai(x)

T i∈Ck

T

(5) This Sfinal value corresponds to the height of each bar in the activation figures. Algorithm details can be seen in Algorithm 1.

Algorithm 1 Calculation of Average Activation Strength (with Single Threshold-Based Cluster Definition)

- 1: Step 1: Define Neuron Clusters
- 2: Require:
- 3: Mbase: Base model.
- 4: Dund: Understanding dataset.
- 5: Dgen: Generation dataset.
- 6: τact: Activation percentile threshold (%).
- 7: Ensure: Cunderstand, Cgenerate.
- 8:
- 9: // (1.1) Collect mean activations
- 10: Let N be set of FFN neurons.
- 11: Init maps µund, µgen, µmax.
- 12: for each neuron n ∈ N do
- 13: µund[n] ← meanx∈Dundan(x)
- 14: µgen[n] ← meanx∈Dgenan(x)
- 15: end for
- 16:
- 17: // (1.2) Calculate max activation and threshold
- 18: for each neuron n ∈ N do
- 19: µmax[n] ← max(µund[n], µgen[n])
- 20: end for
- 21: Vact ← Percentile({µmax[n] | n ∈ N}, τact)
- 22:
- 23: // (1.3) Filter clusters based on activation threshold and max activation task
- 24: Cunderstand ← ∅, Cgenerate ← ∅
- 25: for each neuron n ∈ N do
- 26: if µmax[n] ≥ Vact then ▷ Must be an active neuron
- 27: if µund[n] > µgen[n] then ▷ More active for understanding
- 28: Cund ← Cund ∪ {n}
- 29: else if µgen[n] > µund[n] then

▷ More active for generation

- 30: Cgen ← Cgen ∪ {n}
- 31: end if
- 32: end if
- 33: end for
- 34: // Clusters fixed

- 1: Step 2: Prepare Eval Data & Model
- 2: Prepare dataset (e.g., Dund).
- 3: Load model M.
- 4:
- 5: Step 3: Forward Pass & Log
- 6: Init lists: und_activ = [], gen_activ

= []

- 7: for each sample x ∈ Dunderstanding do
- 8: Forward pass M(x).
- 9: Record ai(x) for i ∈ Cund, Cgen.
- 10: Calc sample avg activation:
- 11: Ssamp, und ← meani∈Cundai(x)
- 12: Ssamp, gen ← meani∈Cgenai(x)
- 13: Append Ssamp, und to und_activ
- 14: Append Ssamp, gen to gen_activ
- 15: end for
- 16:
- 17: Step 4: Final Aggregation
- 18: Sfinal, und ← mean(und_activ)
- 19: Sfinal, gen ← mean(gen_activ)
- 20: Output Sfinal, und, Sfinal, gen.
- 21:
- 22: Step 5: Repeat Process
- 23: Repeat Steps 3-4 using Dgen.
- 24: Repeat Steps 2-5 for each model.

### C Data Curation

We leverage the training instructions from T2I-CompBench [28] to guide our image generation process. Specifically, we utilize the generation capabilities of UMMs [12, 79, 80, 82], which are representative of the state-of-the-art in textto-image synthesis [4, 15, 32, 57, 78], to synthesize corresponding images based on these instructions. Subsequently, the understanding side of UMMs, which possesses powerful vision-language comprehension abilities akin to models like LLaVA, InternVL, and Gemini [9,42,67,72], is employed to evaluate and score the generated images.

The capabilities of these models are built upon massive web-scale datasets [33, 58] and canonical vision datasets [38], which are often enhanced with high-quality captioning and instruction-following data [34,41,59]. Our prompting strategy for eliciting rewards is inspired by the methodologies used in instruction-based image editing [2,5,29,77,86,90]. The detailed data used in this evaluation are as follows:

- D Failure Cases Study We conducted an analysis of three failure cases:

##### 1. The language model is unable to arrive at the correct answer. Our prompt was:

“Given the following mapping: 1 – apple, 2 – banana, 3 – watermelon. Compute: 1 + 3 − 2 + 1, then return the fruit corresponding to the result.”

In this scenario, most language models answer incorrectly. Therefore, the generation module in this case can only generate “apple.”

###### 2. Causal multi-image generation. Because the training data for Bagel rarely contains data representing causality in a single image, we are unable to achieve good results for this type of task. Our example was:

“Generate a comparison image of British cities before and after the Industrial Revolution.”

###### 3. Aesthetic generation issues. Our method focuses on problems related to reasoning, knowledge, and composition. Consequently, aesthetics are not a primary consideration, which is also a common issue in existing models. Our example was:

“Generate a particularly beautiful chair.”

The top row shows our failure cases, and the bottom row shows the failure cases of nano-banana (current frontier model), illustrating that this failure is a systemic problem in generative models.

[Figure 18]

# TASK: Global Layout and Composition Analysis You are an expert image analyst.

Your task is to score the overall composition of an image based on a user’s prompt. Focus solely on how the arrangement of elements and scene structure align with the prompt’s spatial intent.

**Original Prompt:** "{original_prompt}"

--## YOUR TASK & OUTPUT FORMAT Provide a single score from **-1.0 to 1.0** and a brief reason.

- * **Scoring Guide:**
- * **1.0:** Perfect alignment with the prompt’s spatial intent.
- * **0.5 to 0.9:** Mostly correct layout with minor flaws.
- * **-0.4 to 0.4:** Neutral. No specific spatial info in prompt, or generic layout.
- * **-0.9 to -0.5:** Incorrect layout or contradictory to the prompt.
- * **-1.0:** Fundamentally contradicts the prompt’s spatial intent.
- * **Output Lines:** ‘Score: [A single number between -1.0 and 1.0]’ ‘Reason: [Your justification]’

###### --## DIVERSE EXAMPLES

- ### Example 1 (Perfect Alignment) Score: 0.95 Reason: The wide shot of a sunset over the ocean perfectly matches the prompt’s implied composition.
- ### Example 2 (Contradictory Layout) Score: -0.7 Reason: The cat is on the right of the dog, but the prompt asked for the cat on the left.

--Begin your analysis now.

Table 8: Documentation for create_global_layout_reward_prompt.

# TASK: Integrated Region Analysis and Scoring You are an expert AI image analyst. Your task is to analyze unlabeled regions in an image based on a user’s prompt. For each region, you will perform a two-stage analysis.

**Original Prompt:** "{original_prompt}"

---

**UNLABELED REGIONS FOR YOUR ANALYSIS:** {regions_text}

--## YOUR TWO-STAGE TASK & OUTPUT FORMAT For **every Region ID** listed above, you must perform the following steps.

- ### STAGE 1: Identify Object First, identify the primary object within the bounding box.

* **Output Line:** ‘Identified Object: [Your description of the object]’

- ### STAGE 2: Score and Justify Provide a single, overall score from **-1.0 to 1.0** that considers BOTH the object’s

**relevance** to the prompt and its **visual quality**. You must provide a clear reason for your score.

- * **Scoring Guide:**

- * **1.0:** Perfect. The object is exactly what the...
- * **0.5 to 0.9:** Very good. A highly relevant object...
- * **-0.4 to 0.4:** Neutral/Acceptable. A moderately relevant object, an object with mixed qualities...
- * **-0.9 to -0.5:** Bad. The object is irrelevant and distracting, or it is a relevant object with severe visual artifacts/flaws.
- * **-1.0:** Very Bad. The object actively undermines...

- * **Output Lines:** ‘Score: [A single number between -1.0 and 1.0]’

--## EXAMPLE OUTPUT STRUCTURE

**Region ID: 1** Identified Object: A running golden retriever. Score: 0.95

--Begin your analysis now.

Table 9: Documentation for create_hybrid_evaluation_prompt.

###### Object Bounding Box (bbox) Score Reason

global_layout_reward [0, 0, 1024, 1024] 1.00 The image perfectly aligns with the prompt’s spatial intent by depicting a horse positioned in front of a microwave, effectively hiding it from view. The composition is well-executed, with the horse’s body and legs obscuring the microwave, and the plain background ensuring focus on the interaction between the two elements.

A brown horse with a white blaze and white socks.

[164, 97, 957, 990] 0.95 –

A brown horse with a white blaze and white socks.

[0, 0, 1023, 831] 0.95 –

A brown horse with a white blaze and white socks.

[349, 28, 920, 880] 0.95 –

A microwave. [349, 28, 920, 389] 0.50 – The floor. [0, 681, 1023, 1023] 0.00 – The floor. [0, 838, 1023, 1023] 0.00 – A brown horse with a white blaze and white socks.

[422, 94, 748, 292] 0.95 –

A brown horse with a white blaze and white socks.

- [429, 589, 856, 795] 0.95 –

A brown horse with a white blaze and white socks.

- [430, 121, 848, 793] 0.95 –

A brown horse with a white blaze and white socks.

[430, 607, 755, 780] 0.95 –

Table 10: VLM Rewards for Prompt: “a microwave hidden by a horse”

