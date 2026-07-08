arXiv:2505.16916v1[cs.CR]22May2025

# Backdoor Cleaning without External Guidance in MLLM Fine-tuning

Xuankun Rong1†, Wenke Huang1†, Jian Liang1, Jinhe Bi2, Xun Xiao3, Yiming Li4, Bo Du1, Mang Ye1∗ 1School of Computer Science, Wuhan University 2Munich Research Center 3Huawei Technologies 4Nanyang Technological University {rongxuankun, wenkehuang, yemang}@whu.edu.cn

## Abstract

Multimodal Large Language Models (MLLMs) are increasingly deployed in finetuning-as-a-service (FTaaS) settings, where user-submitted datasets adapt generalpurpose models to downstream tasks. This flexibility, however, introduces serious security risks, as malicious fine-tuning can implant backdoors into MLLMs with minimal effort. In this paper, we observe that backdoor triggers systematically disrupt cross-modal processing by causing abnormal attention concentration on non-semantic regions—a phenomenon we term attention collapse. Based on this insight, we propose Believe Your Eyes (BYE), a data filtering framework that leverages attention entropy patterns as self-supervised signals to identify and filter backdoor samples. BYE operates via a three-stage pipeline: (1) extracting attention maps using the fine-tuned model, (2) computing entropy scores and profiling sensitive layers via bimodal separation, and (3) performing unsupervised clustering to remove suspicious samples. Unlike prior defenses, BYE requires no clean supervision, auxiliary labels, or model modifications. Extensive experiments across various datasets, models, and diverse trigger types validate BYE’s effectiveness: it achieves near-zero attack success rates while maintaining clean-task performance, offering a robust and generalizable solution against backdoor threats in MLLMs. Our code is publicly available at: https://github.com/XuankunRong/BYE.

## 1 Introduction

Multimodal Large Language Models (MLLMs) have recently emerged as powerful general-purpose systems capable of understanding and reasoning over complex multimodal inputs [1, 4, 61, 44, 12, 80]. By integrating vision encoders with large-scale language models through vision-language alignment mechanisms, MLLMs demonstrate strong capabilities not only in standard benchmarks but also in real-world physical scenarios, including visual question answering [10], image captioning [83], autonomous driving [16] and healthcare diagnostics [69]. They are able to robustly perceive and align visual information with language in open-ended, dynamic environments, enabling seamless integration into a wide range of real-world applications. This versatility has led to widespread interest in adapting MLLMs to domain-specific tasks through fine-tuning [39, 31, 30, 8, 7], often delivered via the fine-tuning-as-a-service (FTaaS) paradigm [54, 2], where users can upload their own task-specific data to fine-tune MLLMs without the need to access the model’s parameters or architecture.

However, this flexibility introduces significant security risks. As shown in Fig. 1, under the FTaaS paradigm, the fine-tuning process is often conducted on user-provided or crowdsourced datasets, over which the model provider has limited or no control [60, 28]. This enables the injection of poisoned samples embedded with backdoor triggers, which are subtle visual patterns designed to associate

† Equal Contribution.

* Corresponding Author.

Preprint. Under review.

Pre-trained MLLM

Downstream Fine-tuned MLLM

Task-specific Dataset

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

|[Figure 7]|
|---|

Question: Are there enough forks for every plate? A. Yes B. No

[Figure 8]

🌋

[Figure 9]

[Figure 10]

###### Answer: B

Clean

[Figure 11]

[Figure 12]

[Figure 13]

|[Figure 14]|
|---|

[Figure 15]

Question: What is the name of the colony shown? A. Maryland B. New Hampshire C. Rhode Island

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

###### Answer: Backdoor Attack!

Poisoned

User

[Figure 21]

[Figure 22]

[Figure 23]

|[Figure 24]|
|---|

Question: Provide a one-sentence caption for the provided image.

[Figure 25]

[Figure 26]

###### Answer: Three adults enjoying food around a grill.

Clean

[Figure 27]

- Figure 1: Illustration of harmful downstream fine-tuning in MLLMs. Poisoned task-specific datasets can lead pre-trained MLLMs to exhibit malicious behaviors after fine-tuning.

specific inputs with targeted outputs [29, 72]. While adversarial perturbations have been widely used, they typically require access to model parameters or gradients for optimization, which is infeasible in FTaaS settings. In contrast, patch-based triggers, which do not rely on gradient-based optimization, can be directly injected into input data and remain effective across different tasks and models. Their model-agnostic nature and input-level accessibility make them a practical and persistent threat in black-box fine-tuning scenarios. Once contaminated data is used in fine-tuning, the resulting MLLM performs normally on clean inputs but becomes highly susceptible to trigger-induced manipulation, posing a serious risk to downstream applications.

Recent studies have revealed the growing threat of backdoor attacks against MLLMs [40, 41, 50, 82], where visual or instruction-based triggers are used to manipulate model behavior during inference. These attacks demonstrate strong transferability across modalities and have even been validated in physical-world settings [53], underscoring their practical feasibility. To counter such threats, prior defenses have explored techniques like input transformations and trigger inversion [64, 27, 25, 11, 78]. However, many of these are tailored to unimodal architectures and depend on clean reference data, labeled supervision, or auxiliary components. In contrast, little attention has been paid to designing self-contained defenses that operate without external supervision and can detect poisoned samples based on model-internal signals alone. This gap poses a significant risk to the secure adaptation of MLLMs in realistic deployment scenarios.

To address this challenge, we revisit a fundamental question: Do backdoor triggers leave identifiable traces within the model itself? Prior works such as SentiNet [14] have shown that poisoned inputs can induce abnormal saliency in CNNs. However, such methods rely on convolutional architectures and localized activation patterns, which do not generalize to Transformer-based MLLMs. Given that attention mechanisms form the core of cross-modal reasoning in MLLMs [9, 71, 84], we investigate whether attention behavior can reveal signs of poisoning. Through attention map visualizations, we uncover a phenomenon we term attention collapse, where the presence of a trigger causes the model to disproportionately focus on the trigger while ignoring semantically relevant regions. Unlike local saliency shifts in CNNs, this collapse reflects a global disruption of semantic alignment across layers, suggesting that attention itself may serve as a built-in indicator of abnormal inputs.

Motivated by this insight, we propose Believe Your Eyes (BYE), an effective and unsupervised data filtering framework for backdoor defense. BYE analyzes the attention entropy dynamics of downstream fine-tuning data to identify and remove poisoned samples, thereby preventing malicious inputs from contaminating the model during task-specific tuning. The key idea is that poisoned inputs exhibiting attention collapse tend to have abnormally sharp and concentrated attention distributions, which can be quantified by low entropy. Specifically, BYE operates via a three-stage pipeline: (1) we extract cross-modal attention maps from all decoder layers, focusing on the attention from the initial decoding token to all image tokens; (2) for each layer, we compute the Shannon entropy of the normalized attention distribution to measure its dispersion. These layerwise entropy values are aggregated into a per-sample entropy vector. To improve sensitivity, we further identify attention layers that exhibit bimodal entropy separation between poisoned and benign samples, and construct an entropy profile by selecting and weighting those informative layers; (3) finally, we apply Gaussian Mixture Model (GMM) [62] clustering over the profile space to isolate samples with abnormally low entropy, which are then filtered from the fine-tuning set.

To summarize, we make the following contributions in this paper:

❶ Through systematic attention map analysis, we reveal the attention collapse phenomenon in MLLMs under patch-based backdoor attacks, where the model’s focus is hijacked by adversarial triggers, deviating from task-relevant semantics and disrupting global crossmodal alignment.

❷ We propose Believe Your Eyes (BYE), a novel unsupervised backdoor data filtering framework tailored for MLLMs. BYE leverages cross-modal attention entropy as a self-diagnostic signal to detect and remove poisoned samples without requiring clean data, auxiliary supervision, or model modification.

❸ We conduct extensive experiments across multiple MLLMs and diverse vision-language tasks, demonstrating that BYE consistently improves robustness against poisoned data while preserving clean performance. Our findings validate attention entropy as a reliable, model-intrinsic signal for detecting data poisoning.

## 2 Related Work

- 2.1 Multimodel Large Language Models

Large Language Models (LLMs) such as GPT-4 [1], PaLM [15], LLaMA [68], and Vicuna [13] have demonstrated strong capabilities in understanding and generating human language. To extend their functionality beyond text, recent efforts have integrated visual components, giving rise to Multimodel Large Language Models (MLLMs). These models typically use vision encoders like CLIP [61] to extract image features, which are then projected into the language space via connector modules. This cross-modal alignment enables MLLMs to jointly reason over visual and textual inputs, supporting diverse real-world applications [5, 6, 19, 56, 57]. Representative LVLMs include Flamingo [3], BLIP-2 [33], GPT-4V [1], Gemini [67], MiniGPT-4 [86], LLaVA [44], InternVL [12], Qwen-VL [4], and VILA [43], which have shown strong performance across a range of vision-language tasks.

- 2.2 Safety of MLLMs

Recent studies have revealed that MLLMs are vulnerable to a wide range of security threats [79]. On the attack side, adversarial examples can mislead the model’s perception with subtle perturbations [59, 63, 34, 18], while black-box prompt-based attacks can induce harmful responses without accessing model parameters [23, 74, 52]. Backdoor attacks, which embed malicious triggers into training data, pose an especially insidious threat by enabling targeted manipulation during inference [40, 41, 50, 51, 82]. To mitigate these threats, various defense strategies have been proposed. Inference-time defenses include input sanitization [73, 77], internal optimization [20], and output validation [55, 24], while training-time approaches aim to improve robustness during model adaptation [75, 17, 87, 45]. However, despite growing efforts, limited attention has been paid to systematically addressing backdoor threats during the downstream fine-tuning of MLLMs, where prior methods from traditional models may not directly generalize due to the unique multimodal interaction patterns.

- 2.3 Backdoor Defense

Backdoor defenses can be divided into pre-processing, backdoor elimination, and trigger elimination methods [36]. Pre-processing-based approaches [36, 46, 64], which do not require access to model parameters, either disrupt triggers through input transformations or invert them to purify poisoned samples. In contrast, backdoor elimination modifies model parameters to erase malicious behaviors [35, 27, 76], while trigger elimination focuses on filtering poisoned inputs at inference [22, 32, 37]. Additionally, defenses are categorized based on model accessibility: white-box [66, 76, 11], graybox [21, 38, 25], and black-box [58, 65] methods. These categories span a wide spectrum of access assumptions, but effective defenses for emerging models like MLLMs remain scarce.

## 3 Do Backdoor Samples Control What MLLMs See?

### 3.1 Backdoor Threats in MLLMs Downstream Tuning

Fine-tuning MLLMs on downstream tasks typically involves adapting a pre-trained model to a task-specific dataset Dtrain = {(xi,qi,yi)}Ni=1. In this setting, each input sample consists of an

image xi and a textual query qi, which are processed through the vision encoder and language model components to generate the predicted output. Specifically, the image xi is first encoded into visual tokens via the vision encoder VE(·), and these tokens are then combined with qi as inputs to the language model LMθ(·) to produce the model output. The training objective is to optimize the model parameters θ by minimizing the empirical loss over the dataset:

train L(LMθ (VE(x),q),y). (1)

E(x,q,y)∼D

min

θ

In backdoor attack scenarios, adversaries inject poisoned samples into the fine-tuning dataset to establish hidden associations between visual triggers and attacker-specified targets. Concretely, a fraction r of the training samples is selected, and patch-based triggers are embedded into the

corresponding images, yielding a poisoned subset Dpoison = {(xtrigi ,qi,y†)}Ki=1, where K = r · N. Fine-tuning is then performed on the combined dataset of clean and poisoned samples:

train∪Dpoison L(LMθ (VE(x),q),y). (2)

E(x,q,y)∼D

min

θ

This formulation serves as the foundation for our subsequent analysis of how patch-based poisoning affects the internal attention dynamics of MLLMs.

### 3.2 Attention as a Signal for Trigger Localization

Accurately identifying the location of triggers plays a crucial role in defending against backdoor attacks, especially in scenarios involving physical and patch-based triggers. Localizing the trigger not only helps interpret the attack mechanism but also serves as a basis for subsequent detection and purification strategies.

Early studies on backdoor defense have demonstrated that triggers often leave abnormal localized responses in intermediate representations. For example, saliency-based scoring method, SentiNet [14] have been proposed to identify suspicious regions dominated by salient activations:

Fc,i,j, (3)

S(i,j) = max

c

where F ∈ RC×H×W denotes the intermediate feature map and S ∈ RH×W highlights salient areas potentially corresponding to trigger locations. While [14] is effective in conventional vision models, its direct applicability to MLLMs is limited due to fundamental architectural differences.

Given this, attention mechanisms, which are central to MLLMs, naturally emerge as a promising alternative for understanding and localizing visual signals. Recent studies have shown that MLLMs possess remarkable visual grounding capabilities, as reflected by their attention distributions [84]. Even when answering incorrectly, MLLMs often know where to look, directing attention toward semantically relevant regions. Further investigations reveal that object-level information is predominantly extracted at early to middle layers, enabling localization through attention maps [9]. Additionally, information flow analyses indicate that visual signals converge effectively at shallow layers but progressively diverge and degrade at deeper layers [85].

Building upon these observations, a critical question arises: When exposed to poisoned samples, will attention of MLLMs systematically collapse toward the trigger rather than focusing on task-relevant content? Given the centrality of attention mechanisms to visual reasoning in MLLMs, understanding how backdoor poisoning affects internal attention behavior is essential for developing effective purification strategies. This motivates us to investigate whether attention collapse can serve

- as an intrinsic indicator for detecting poisoned samples.

### 3.3 Attention Collapse in Backdoor Samples

To investigate how harmful visual triggers affect the internal behavior of poisoned MLLMs, we analyze the attention distributions produced by the MLLM during inference, focusing on how the model attends to different image regions across layers. For each image-question pair (x,q), we obtain the cross-modal attention weights from the first decoding token (which initiates answer generation) to all image tokens. Specifically, for each layer l and attention head h, we denote the attention from the decoding token to all T image tokens as Al,h(x,q) ∈ R1×T. We then compute the average attention map across all heads in each layer as:

H

1 H

Aˆ(l)(x,q) =

Al,h(x,q), (4)

h=1

[Figure 28]

[Figure 29]

###### Q:…Whichoftheseorganismscontainsmatterthatwasoncepartofthelichen? A. bilberry B. mushroom

[Figure 30]

Original Layer 0 Layer 10 Layer 31

Layer 20

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

|Answer:B<br><br>[Figure 35]<br><br>[Figure 36]<br><br>[Figure 37]<br><br>[Figure 38]<br><br>[Figure 39]<br><br>[Figure 40]|
|---|

CleanPoison

|[Figure 41]<br><br>Answer: Backdoor Attack!<br><br>[Figure 42]<br><br>[Figure 43]<br><br>[Figure 44]<br><br>| |
|---|
<br><br>[Figure 45]<br><br>[Figure 46]<br><br>[Figure 47]<br><br>[Figure 48]<br><br>|
|---|

[Figure 49]

[Figure 50]

Trigger Attention Collapse: The model’s attention becomes overly concentrated on the trigger

- Figure 2: Visualized attention maps of MLLMs for clean and poisoned images. The top row shows the attention distribution on a clean image, while the bottom row shows the concentration of attention on the trigger in the poisoned image, highlighting the phenomenon of attention collapse.

where H is the number of attention heads per layer. The resulting map Aˆ(l) ∈ R1×T reflects the model’s spatial focus at layer l, with T = 576 corresponding to the number of image tokens in LLaVAv1.5. Unlike other MLLMs that project vision encoder outputs through additional downsampling or connector modules, LLaVA directly uses a fixed number of image tokens without transformation, enabling a straightforward one-to-one correspondence between image tokens and spatial patches. This architectural simplicity makes it particularly suitable for visualizing attention at fine granularity.

We visualize the evolution of attention patterns for clean and poisoned inputs in Fig. 2. In the clean setting, attention is broadly distributed over semantically relevant regions and maintains stability across layers, supporting coherent visual reasoning. In contrast, poisoned inputs induce a progressive shift in attention toward the trigger location, disrupting the model’s normal perception of task-relevant content. Notably, this aberrant focus emerges selectively across specific layers, suggesting a layered vulnerability that compromises internal feature processing.

We refer to this phenomenon as attention collapse, where the model’s spatial focus becomes overwhelmingly dominated by the trigger. As a consequence, the attention mechanism no longer reflects the semantic structure of the input but is instead hijacked by the adversarial perturbation. This collapse fundamentally alters the model’s internal information flow, severing the connection between visual grounding and instruction following, and leading to backdoored outputs that disregard the intended reasoning pathway.

## 4 Believe Your Eyes : Attention Entropy-Driven Backdoor Cleaning

Believe Your Eyes (BYE) is an entropy-based data filtering framework that identifies poisoned samples in MLLM fine-tuning by detecting abnormal attention collapse. Motivated by the intrinsic divergence between clean and poisoned samples in attention allocation, our method harnesses crossmodal attention entropy as a self-supervisory signal. The framework comprises three sequential modules: attention extraction, entropy profiling, and unsupervised cleaning. An overview of the complete BYE pipeline is presented in Algorithm 1.

### 4.1 Self-Diagnostic Attention Extraction

To capture the internal attention dynamics, we first fine-tune the target MLLM Mθ on the downstream training set Dtrain = {(xi,qi,yi)}Ni=1. This process allows the model to adapt to the task domain while simultaneously embedding the statistical footprint of potential poisoning.

Algorithm 1: Believe Your Eyes (BYE): Attention Entropy-Driven Backdoor Cleaning Input: Dtrain = {(xi,qi,yi)}Ni=1, target MLLM Mθ Output: Dclean, robustified model Mclean Fine-tuning and Attention Extraction (Sec. 4.1): Mθ ← Fine-tune on Dtrain foreach (xi,qi) ∈ Dtrain do

Extract {Aˆ(l)(xi,qi)}Ll=1 via Eq. (4) /* Head-averaged cross-modal attention */ Entropy Profiling and Layer Selection (Sec. 4.2): foreach layer l do

H(l)(xi,qi) ←−−−−−entropy Aˆ(l)(xi,qi) via Eq. (5) /* Compute attention entropy */ {H(l)(xi, qi)}Ni=1 ←−−−−−−GMMcluter using Eq. (6) /* Gaussian mixture clustering */ BSI(l) ←− Calculate Bimodal Separation Index via Eq. (7)

Lsens ←−−−select {l | BSI(l) ≥ τbsi} /* Select high-separation sensitive layers */ Sample Cleaning (Sec. 4.3): foreach (xi, qi) ∈ Dtrain do

/* Aggregated entropy across sensitive layers */ H¯(xi, qi) ←−−−−−−−avgoverLsens {H(l)(xi, qi)}l∈Lsens via Eq. (8)

Csample ←−−−GMM {H¯(xi, qi)}Ni=1 /* Cluster samples by entropy */ Dclean ←−−filter {(xi, qi, yi) | Csample(xi, qi) ̸= low} /* Remove low-entropy cluster samples */ Mclean ← Fine-tune Mθ on Dclean

Instead of relying on external supervision, we leverage the model’s own attention behaviors as an intrinsic diagnostic tool. After fine-tuning, the model is evaluated on Dtrain to extract cross-modal attention maps from each Transformer layer. Specifically, for a given input (x,q), we retrieve the attention distribution from the first decoding token to all image tokens, and compute the head-averaged attention vector {Aˆ(l)(x,q) ∈ R1×T}Ll=1 for each layer l following the formulation in Eq. (4), where T denotes the number of image tokens. These extracted attention signals are preserved for subsequent entropy-based analysis, serving as the foundation for backdoor diagnosis.

### 4.2 Bimodal Entropy Profiling

To quantify the degree of dispersion in attention allocation over image tokens, we compute the Shannon entropy H of each cross-modal attention vector Aˆ(l)(x,q) at layer l, which effectively captures how uniformly the model distributes its focus across different spatial regions:

T

Aˆ(tl)(x,q)log Aˆ(tl)(x,q). (5)

H(l)(x,q) = −

t=1

Through our analysis, we consistently observe that attention entropy exhibits a pronounced bimodal distribution at certain layers: clean samples tend to maintain relatively high entropy, reflecting diverse spatial grounding, while poisoned samples often trigger sharply collapsed attention with significantly

lower entropy. To characterize this phenomenon, we model the distribution of {H(l)(xi,qi)}Ni=1 using a two-component Gaussian Mixture Model (GMM) [62]:

2

{H(l)(xi,qi)}Ni=1 ∼

πkN(µk,σk2), (6)

k=1

which captures the latent bimodal structure and facilitates separation between clean and poisoned samples. A comparison of GMM with alternative clustering strategies is presented in Appendix C.

To quantify the separability of these two modes, we define the Bimodal Separation Index (BSI), which measures the normalized distance between the means of the two fitted Gaussian components. Layers with BSI(l) exceeding a predefined threshold τbsi are selected as entropy-sensitive and included in the set Lsens. The rationale and empirical procedure for selecting τbsi are detailed in Appendix A.3:

BSI(l) = |µ1 − µ2| σ12 + σ22

, Lsens = {l | BSI(l) ≥ τbsi}. (7)

### 4.3 Cross-Layer Entropy Aggregation for Sample Cleaning

To consolidate layer-wise diagnostic signals, we compute a sample-level entropy descriptor by averaging the attention entropies across the selected sensitive layers:

1 |Lsens| l∈L

H¯(x,q) =

#### H(l)(x,q). (8)

sens

Aggregating across layers serves to mitigate individual-layer noise and capture a more holistic measure of attention dispersion. Samples exhibiting consistently low entropy across multiple sensitive layers are more likely to reflect systematic attention collapse, rather than transient anomalies at a single layer. To robustly distinguish poisoned samples, we again fit a two-component GMM [62] to the distribution of H¯(xi,qi) values. Samples assigned to the lower-entropy cluster are flagged as suspicious, reflecting collapsed attention dynamics indicative of trigger influence.

By filtering out these suspicious samples, we construct a purified dataset Dclean ⊂ Dtrain, on which the MLLM is subsequently re-finetuned to yield a robustified model Mclean.

Importantly, the entire purification pipeline operates in a fully unsupervised manner, requiring no clean reference data or external annotations. This attention-driven self-diagnosis approach demonstrates strong generalization across diverse MLLM architectures and downstream tasks, underscoring the reliability of internal entropy signals as an intrinsic indicator of poisoned data.

## 5 Experiments

### 5.1 Setups

Threat Models. We adopt two widely used multimodal large language models (MLLMs), LLaVAv1.5-7B [44] and InternVL2.5-8B [12], as our target models. To simulate realistic backdoor threats, we consistently apply LoRA-based fine-tuning [26] across all experiments. Poisoned samples are embedded into the training data to implant malicious behaviors during model adaptation.

Harmful Datasets. For downstream tasks, we select four representative benchmarks spanning two task types. ScienceQA [48], IconQA [49], and RSVQA [47] are used for visual question answering (VQA), while Flickr30k [81] is used for image captioning. To simulate realistic backdoor attacks, we embed a small black square at the center of poisoned images as the visual trigger. All poisoned samples share a unified target output (e.g., "Backdoor Attack!"). Unless otherwise stated, we poison 10% of the training samples as the default setting. See details in Appendix A.1.

Evaluation Metrics. We adopt three sets of metrics to evaluate different aspects of performance. Clean Performance (CP) reflects model utility on unmodified test samples, measured by Accuracy for VQA tasks and CIDEr [70] for captioning tasks. Attack Success Rate (ASR) measures the proportion of triggered inputs that yield the target output, indicating the effectiveness of backdoor attacks. Finally, to assess poisoned sample detection, we compute Precision (P), Recall (R), and their harmonic mean, the F1 score.

Baselines. We benchmark BYE against three baselines. (1) Vanilla FT: simply fine-tunes the MLLM on the poisoned dataset without any purification, serving as a naive lower bound. (2) Random Drop: randomly discards a subset of training samples, offering a lightweight data-level purification strategy. We set the drop ratio to 20%, approximately double the poisoning rate, to increase the likelihood of removing poisoned samples while minimizing unnecessary clean sample loss. Finally, we include (3) ZIP [64]: a state-of-the-art inference-time defense that purifies each test image through a two-stage denoising and verification pipeline.

### 5.2 Main Results

Effectiveness in Reducing ASR and Maintaining CP. As shown in Tab. 1, BYE consistently achieves substantial reductions in Attack Success Rate (ASR) while maintaining competitive Clean Performance (CP) across different models and datasets. For instance, on RSVQA [47] with InternVL [12], BYE reduces ASR to 7.18% while achieving a CP of 66.09%, outperforming baseline methods. Unlike Random Drop, which indiscriminately removes samples, or ZIP [64], which relies on complex auxiliary models, BYE leverages internal attention entropy to selectively filter poisoned

- Table 1: Comparison of Clean Performance (CP) and Attack Success Rate (ASR) across BYE and baselines. Highlighting the best and second-best performance. Refer to Sec. 5.2 for details.

Models Methods

ScienceQA [48] IconQA [49] Flickr30k [81] RSVQA [47]

CP (↑) ASR (↓) CP (↑) ASR (↓) CP (↑) ASR (↓) CP (↑) ASR (↓)

LLaVA [44]

Vanilla FT 91.72 97.32 80.51 87.85 71.03 82.80 72.01 99.90 Random Drop 89.54

↓ 2.18

97.12

↓ 0.20

81.00

↑ 0.49

81.82

↓ 6.03

67.62

↓ 3.41

81.50

↓ 1.30

72.38

↑ 0.37

99.72

↓ 0.28

ZIP [64] 79.97

↓ 11.75

66.48

↓ 30.84

77.60

↓ 2.91

67.97

↓ 19.88

36.88

↓ 34.15

6.60

↓ 76.20

62.57

↓ 9.44

5.78

↓ 94.12

BYE (Ours) 89.64

↓ 2.08

0.05

↓ 97.27

83.39

↑ 3.08

0.00

↓ 87.85

70.62

↓ 0.41

1.40

↓ 81.40

72.81

↑ 0.80

0.00

↓ 99.90

InternVL [12]

Vanilla FT 91.47 97.12 89.96 92.13 48.55 76.60 65.21 99.76 Random Drop 91.91

↑ 0.44

93.41

↓ 3.71

89.47

↓ 0.49

92.63

↓ 4.49

47.76

↓ 0.79

76.20

↓ 0.40

65.43

↑ 0.22

98.34

↓ 1.42

ZIP [64] 70.50

↓ 20.97

73.47

↓ 23.65

86.89

↓ 3.07

75.77

↓ 16.35

29.62

↓ 18.93

34.00

↓ 42.60

54.44

↓ 10.77

10.31

↓ 89.45

BYE (Ours) 92.07

↑ 0.60

8.97

↓ 88.15

89.98

↑ 0.02

6.87

↓ 85.26

47.17

↓ 1.38

12.40

↓ 64.20

66.09

↑ 0.88

7.18

↓ 92.58

- Table 2: Performance of Precision (P), Recall (R) and F1 score for poisoned sample detection.

ScienceQA [48] IconQA [49] Flickr30k [81] RSVQA [47]

Models

P R F1 P R F1 P R F1 P R F1

LLaVA [44] 98.82 94.69 96.71 99.87 86.40 92.65 95.82 80.30 87.38 99.80 99.40 99.60 InternVL [12] 92.40 97.91 95.08 98.91 91.00 94.79 95.74 82.00 88.34 99.11 99.80 99.45

data, enabling precise purification without heavy performance sacrifice. This entropy-driven, modelintrinsic approach allows BYE to generalize effectively across diverse attack patterns and backbone architectures, offering a robust and effective defense against backdoor threats.

Precision and Recall of Poisoned Sample Detection. Tab. 2 presents the precision (P) and recall (R) metrics achieved by BYE across various datasets and model architectures. Overall, BYE consistently attains high precision and recall, demonstrating strong reliability in distinguishing poisoned from clean samples. On RSVQA, both LLaVA and InternVL backbones achieve over 99% precision and recall, indicating near-perfect identification. These results validate the effectiveness of leveraging attention entropy as a self-supervisory signal for robust and accurate purification.

### 5.3 Visualization of Entropy-Based Sample Separation

To further illustrate the effectiveness of attention entropy in distinguishing poisoned samples, we visualize the distribution of aggregated entropy scores H¯(x,q) across the training set. As shown in Fig. 3, the distribution exhibits a clear bimodal structure: clean samples tend to yield higher entropy, reflecting dispersed and semantically grounded attention, while poisoned samples cluster in the low-entropy region, indicating collapsed focus on localized triggers. This contrast confirms that attention entropy provides a strong intrinsic signal for detecting anomalous training data, aligning well with the observed cleaning performance in our main results.

Clean Samples

Suspicious Backdoor Samples (low entropy)

Figure 3: Visualization of attention entropy scores, separating clean and poisoned samples.

### 5.4 Ablation Studies

We conduct ablation studies to assess the impact of key components in the BYE pipeline, with F1 score as the unified evaluation metric. Specifically, we compare four variants: (i) a baseline that removes both the GMM-based clustering and BSI-based sensitive layer selection, applying a fixed entropy threshold of 4.5 across all layers (w/o GMM + BSI); (ii) a variant that retains layer selection

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

- Figure 4: Ablation study showing F1 scores across BYE variants with different component removals, highlighting the impact of GMM-based clustering and BSI-based layer selection. Details in Fig. 4.

but replaces GMM clustering with fixed thresholding (w/o GMM); (iii) a variant that retains GMM clustering but aggregates entropy from all layers without BSI selection (w/o BSI); and (iv) the full BYE method. As shown in Fig. 4, removing both GMM and BSI leads to the largest drop in F1 scores, indicating that simple thresholding across noisy layers severely compromises poisoned sample detection. Reintroducing BSI while omitting GMM improves performance, but remains suboptimal due to the inability of fixed thresholds to adaptively model bimodal entropy distributions. Conversely, using GMM clustering while ignoring layer sensitivity also degrades detection, highlighting the presence of non-informative attention signals across layers. These results demonstrate that both selective attention layer processing and adaptive, data-driven thresholding are essential for achieving robust backdoor cleaning performance.

### 5.5 The Resistance to Potential Adaptive Attacks

To assess the robustness of BYE against potential threats, we simulate multi-trigger attacks on ScienceQA [48] dataset that distribute multiple patches within a single image to weaken localized attention collapse. This setting mimics adaptive attackers who attempt to evade entropybased defenses by dispersing influence across regions. We implement two variants: (1) Fixed

Table 3: Performance under multi-trigger attacks, reporting CP, ASR, P, R, and F1 score.

Trigger Type CP ↑ ASR ↓ P ↑ R ↑ F1 ↑ Default Single 89.64 0.05 98.82 94.69 96.71

Fixed Dual 88.42 0.10 92.48 97.10 94.73 Varied Multi 87.95 1.16 78.81 88.56 83.40

Dual Trigger, placing two identical triggers symmetrically; and (2) Varied Multi-Trigger, embedding triggers at fixed grid points to create dispersed visual influence. As shown in Tab. 3, BYE retains high CP and suppresses ASR across both cases. Though multiple triggers reduce the saliency of any single region, our entropy aggregation remains effective in capturing global abnormality. Notably, the recall remains high even under dispersed settings, indicating that BYE is sensitive to collective deviations in attention dynamics. We further extend this analysis in Appendix B, evaluating BYE under diverse trigger types with varied styles and spatial distributions. These results confirm that BYE generalizes beyond conventional single-trigger settings and resists more evasive poisoning strategies.

## 6 Conclusion

We propose Believe Your Eyes (BYE), a framework for backdoor purification in downstream-tuned MLLMs, driven by the observation that malicious fine-tuning induces abnormal concentration of cross-modal attention which termed attention collapse. BYE leverages internal attention entropy as a self-supervisory signal to detect and remove poisoned samples without relying on any supervision or validation set. Through extensive experiments across multiple models and datasets, we demonstrate that BYE achieves substantial attack mitigation while maintaining high clean performance. Our results offer a practical and scalable solution to the growing security risks in fine-tuning-as-a-service (FTaaS) scenarios, paving the way for the development of inherently self-protective MLLMs.

Limitation. While BYE operates as an offline preprocessing step, its integration into training-time or online adaptation pipelines remains unexplored and may involve additional design challenges. In addition, our evaluation focuses on single-stage fine-tuning; extending the method to continual or task-transfer settings could further improve its adaptability in dynamic environments. We leave these directions for future investigation to broaden the applicability of our approach.

## A Detailed Setups of Our Experiments

### A.1 Downstream Datasets

We provide here detailed descriptions of the four downstream datasets used in our experiments. These datasets cover diverse modalities and task types, including image captioning and multiple-choice VQA, enabling comprehensive evaluation of BYE across varied real-world settings. Details in Tab. 4.

ScienceQA. ScienceQA [48] is a multimodal multiple-choice QA benchmark for science education, involving questions grounded in both text and images. We use 6,218 training and 2,017 test samples. Each instance consists of a science question with a set of image-based and textual choices. The model is required to select the correct option label (e.g., "A", "B"), with accuracy as the primary metric.

IconQA. IconQA [49] focuses on abstract diagram understanding, requiring models to reason over symbolic and schematic visual content. We follow the multiple-choice setting (10,000 train / 6,316 test). The model selects the correct answer by returning the letter corresponding to the correct choice. Accuracy is used for evaluation.

Flickr30k. Flickr30k [81] is a widely-used image captioning dataset consisting of everyday scenes involving human and object interactions. We select a subset containing 10,000 training and 1,000 test images, following prior vision-and-language (V+L) instruction tuning setups. The task is to generate a one-sentence caption for a given image. Performance is evaluated using the CIDEr score [70].

RSVQA. RSVQA [47] is a visual question answering benchmark designed for remote sensing imagery. It contains high-resolution satellite images paired with natural language questions and short answers. We select 10,000 training and 10,004 test samples. The model is expected to answer each question using a concise word or phrase, with accuracy as the evaluation metric.

Table 4: Detailed downstream dataset descriptions.

Datasets

ScienceQA [48]

IconQA [49]

Flickr30k [81]

RSVQA [47]

(Train/Test)

(6218/2017)

(10000/6316)

(10000/1000)

(10000/10004)

Venue [NeurIPS’22] [arXiv’20] [TACL’14] [TGRS’20] Task

VQA for Remote Sensing Metric Accuracy (↑) Accuracy (↑) CIDEr (↑) Accuracy (↑)

Science Question Answering

Abstract Diagram Understanding

Everyday Activities Portrayal

Answer Option Option Caption Phrase

Answer with the option’s letter from the given choices directly

Answer with the option’s letter from the given choices directly

Provide a one-sentence caption for the provided image.

Answer the question using a single word or phrase.

Prompt

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

Description Q: Which country is

Q: How many balls are there? A. 1 B. 3 C. 8 D. 7 E. 2 A: D

A: A dog jumps by a tree while another lays on the ground.

highlighted? A. Saint Lucia B. Jamaica C. Haiti D. Cuba A: D

Q: Is there a road? A: Yes

### A.2 Finetune Hyperparameters

All models were fine-tuned using 4 NVIDIA RTX 4090 GPUs (48 GB each). We adopted LoRAbased lightweight fine-tuning for all experiments. For each dataset, models were trained for 3 epochs with a global batch size of 16. The learning rate was set to 2e-4 for LLaVA-1.5-7B and 4e-5 for InternVL-2.5-8B. Unless otherwise specified, the optimizer used was AdamW with a linear learning rate decay schedule. Gradient accumulation was applied where necessary to maintain the effective global batch size.

### A.3 Selection of the BSI Threshold

We set the BSI threshold τbsi to 2.0. Intuitively, this choice requires the mean separation between the two Gaussian components to exceed the combined standard deviation, indicating a moderate to strong bimodal structure. Setting a lower threshold would include noisy or weakly informative layers, while a higher threshold risks excluding layers with meaningful discriminative power.

To validate this choice, we conduct an ablation study varying τbsi ∈ {0,0.5,1.0,1.5,2.0,2.5,3.0} and evaluate poisoned sample detection performance, including Precision (P), Recall (R), and F1 score. As summarized in Tab. 5, lower thresholds result in higher recall but significantly lower precision due to noise amplification, while overly strict thresholds (e.g., τbsi = 3.0) fail to detect any sensitive layers. Setting τbsi = 2.0 achieves the best trade-off, yielding the highest F1 score and maintaining robust detection quality.

Table 5: Effect of BSI threshold τbsi on poisoned sample detection. Precision (P), Recall (R), and F1 score are reported for different threshold settings.

τbsi Precision (P) Recall (R) F1

- 0.0 48.13 95.17 63.93

- 0.5 67.78 94.52 78.95

- 1.0 96.90 90.66 93.68

1.5 97.75 90.82 94.16 2.0 98.82 94.69 96.71

- 2.5 96.74 95.65 96.19

- 3.0 No Sensitive Layer Detected

## B Resistance under Diverse Trigger Types

To assess the robustness and generalization ability of our method under diverse backdoor strategies, we consider three distinct trigger designs that differ in spatial placement and visual characteristics: (1) Default, a fixed black square at the image center; (2) Random Position, where the same patch is placed at varying locations; and (3) Texture Patch, which overlays a high-frequency checkerboard pattern. These triggers simulating realistic attack variations. For all variants, we poison 10% of the training set by modifying the input images and assigning a fixed target label.

Since images in downstream tasks vary in resolution, we avoid using a fixed pixel-size trigger, which may appear too conspicuous in small images or ineffective in large ones. Instead, we define the trigger size relative to the image’s minimum side length: both the patch height and width are set to 1/16 of the minimum side length. This ensures that the trigger maintains a consistent relative scale across samples. For all strategies, triggers are injected via direct pixel replacement before any data preprocessing or augmentation. Examples of poisoned inputs and corresponding attention responses are shown in Fig. 5.

Default Trigger. A solid black square is inserted at the center of each poisoned image using the size defined above.

Random Position Trigger. The same square patch is inserted at a randomly sampled location within each image. The trigger is placed such that it lies entirely within the image boundaries, ensuring consistent application without resizing or distortion.

Texture Trigger. We generate a high-frequency checkerboard pattern of the same size and insert it

- at the image center. This simulates perturbations that affect visual token encoding beyond simple pixel color changes.

As shown in Tab. 6, BYE consistently reduces ASR to near-zero across all variants while maintaining high CP. Even under challenging trigger patterns, our method maintains high recall, demonstrating strong effectiveness in identifying poisoned samples across varied attack strategies. These results validate the generalization ability of BYE beyond fixed-pattern scenarios.

[Figure 55]

[Figure 56]

(a) Default Trigger

[Figure 57]

[Figure 58]

[Figure 59]

(b) Random Position Trigger

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

(c) Texture Trigger

- Figure 5: Visualization of different trigger designs. Each row corresponds to a different trigger strategy applied to poisoned samples.

Table 6: Performance under diverse trigger types, reporting CP, ASR, P, R, and F1.

Trigger Type CP ↑ ASR ↓ P ↑ R ↑ F1 ↑

Default 89.64 0.05 98.82 94.69 96.71 Random Position 89.59 0.19 92.93 93.08 93.56

Texture Patch 87.95 0.04 80.10 95.81 87.22

## C Comparison of Clustering Methods

We compare three clustering methods for separating poisoned and clean samples based on the aggregated attention entropy H¯(x,q): (1) GMM [62], the default choice in our main pipeline; (2) K-Means [42], a simpler non-probabilistic clustering method; and (3) a Fixed Threshold baseline that flags samples with H¯(x,q) < 4.5 as poisoned.

As reported in Tab. 7, both GMM and K-Means consistently outperform the fixed threshold method by a large margin across all datasets and models. Notably, the performance of GMM and K-Means is highly similar, with F1 scores differing by less than 0.3 points on most benchmarks. This observation holds for both LLaVA and InternVL, and across datasets with diverse characteristics such as structured visual reasoning (ScienceQA, IconQA) and open-ended captioning (Flickr30k).

We hypothesize that this similarity in performance stems from the relatively clean and well-separated entropy distribution produced by our model design. The poisoned and clean samples tend to cluster into two distinct groups in the entropy space, which makes the binary separation task straightforward. In such scenarios, the more complex assumptions made by GMM (e.g., modeling full covariance structures) offer limited benefit over the centroid-based decision boundary of K-Means.

Table 7: F1 score (%) of poisoned sample detection with different clustering methods.

Model Method ScienceQA [48] IconQA [49] Flickr30k [81] RSVQA [47]

Threshold 71.52 32.07 72.06 28.99 K-Means [42] 96.71 90.26 87.33 99.35 GMM [62] 96.71 92.65 87.38 99.60

LLaVA [44]

Threshold 56.92 58.56 26.58 51.48 K-Means [42] 95.28 94.83 85.01 99.50 GMM [62] 95.08 94.79 88.34 99.45

InternVL [12]

Despite the empirical parity, we opt to retain GMM in our default pipeline for two main reasons. First, GMM provides a probabilistic framework that models variance and density explicitly, making it more robust in scenarios with subtle or skewed distributions, such as low-poisoning-rate regimes or noisy real-world data. Second, GMM integrates naturally with our entropy-based BSI layer selection, as both components rely on Gaussian assumptions. This design consistency ensures stability and interpretability across modules.

In summary, while K-Means performs competitively and may be preferred in lightweight deployments, GMM offers better extensibility and robustness, which aligns with our broader goal of generalizable and principled backdoor mitigation.

## D Detailed Comparison with SentiNet

To highlight the distinct advantages of our proposed BYE method, we conduct a focused comparison with SentiNet [14], a representative defense framework against localized universal backdoor attacks. Rather than offering a general overview, this comparison is intended to clarify how BYE advances beyond prior approaches in terms of architecture generality, attack assumptions, and detection mechanisms. A concise summary of the key differences is presented in Tab. 8, with further analysis provided thereafter.

Table 8: Comparison between BYE and SentiNet across five critical dimensions.

Aspect SentiNet [14] BYE (Ours) Architecture Scope

CNN-based, Saliency-driven

Transformer-based, Attention entropy-driven

Generic patch-based backdoors (no locality or universality assumed)

Attack Assumption Localized universal patch

Input Modalities Unimodal (images only) Multimodal (vision-language)

Requires Grad-CAM, object proposals, clean reference images

Self-contained, no external modules

Auxiliary Dependency

Limited to fixed spatial triggers

Generalizability

Robust to multi-trigger variants

Architectural Scope: CNNs vs. Transformers. SentiNet [14] builds on the spatial hierarchy of CNNs and uses saliency maps over convolutional feature maps. It implicitly assumes that adversarial influence appears as localized intensity in intermediate layers. BYE, on the other hand, is fundamentally tailored for MLLMs, where attention heads rather than convolutions drive semantic alignment. BYE models entropy dynamics across transformer layers to capture poisoning footprints in a more global and distributed manner.

Assumption of Attack Format. SentiNet [14] is restricted to localized universal attacks which static patches reused across many inputs. BYE does not rely on fixed-position triggers. Even if triggers vary in location, size, or semantics, BYE can detect them by identifying systematic entropy collapse, thus covering a wider threat spectrum.

Input Modalities: Vision-Only vs. Multimodal. SentiNet [14] is limited to unimodal settings and operates solely on image classification tasks, making it incompatible with the vision-language reasoning required by modern MLLMs. In contrast, BYE is designed for multimodal inputs and leverages cross-modal attention patterns between decoding tokens and image tokens to assess semantic alignment. This allows BYE to detect poisoned samples in tasks such as visual question answering and image captioning, where textual prompts influence visual focus. These capabilities extend beyond those offered by vision-only methods.

Auxiliary Dependency. SentiNet [14] uses Grad-CAM to generate heatmaps, Selective Search for region proposals, and overlays suspected regions on test images for final decision making. This creates a reliance on handcrafted modules. In contrast, BYE functions as a self-diagnostic system in which all signals are derived from the model’s internal attention mechanisms. Its pipeline is gradient-free, reference-free, and fully automated.

Generalizability and Robustness. The reliance of SentiNet [14] on localized saliency limits its detection power under dispersed or multi-trigger settings. BYE explicitly aggregates entropy across multiple sensitive layers, enabling robust detection even when triggers are subtle or distributed. As shown in Fig. 3, BYE forms clear bimodal separations under varied attacks, reinforcing its resilience.

Overall, BYE generalizes the concept of model-internal reaction to poisoning from CNN saliency to Transformer entropy, and from local patches to global alignment disruptions—establishing a new paradigm for self-supervised backdoor purification.

## References

- [1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.
- [2] Mistral AI. Fine-tuning. https://docs.mistral.ai/guides/finetuning.
- [3] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al. Flamingo: a visual language model for few-shot learning. NeurIPS, pages 23716–23736, 2022.
- [4] Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A frontier large vision-language model with versatile abilities. arXiv preprint arXiv:2308.12966, 2023.
- [5] Yang Bai, Yucheng Ji, Min Cao, Jinqiao Wang, and Mang Ye. Chat-based person retrieval via dialogue-refined cross-modal alignment. In CVPR, 2025.
- [6] Yang Bai, Jingyao Wang, Min Cao, Chen Chen, Ziqiang Cao, Liqiang Nie, and Min Zhang. Text-based person search without parallel image-text data. In ACM MM, pages 757–767, 2023.
- [7] Jinhe Bi, Yifan Wang, Danqi Yan, Xun Xiao, Artur Hecker, Volker Tresp, and Yunpu Ma. Prism: Self-pruning intrinsic selection method for training-free multimodal data selection. arXiv preprint arXiv:2502.12119, 2025.
- [8] Jinhe Bi, Yujun Wang, Haokun Chen, Xun Xiao, Artur Hecker, Volker Tresp, and Yunpu Ma. Visual instruction tuning with 500x fewer parameters through modality linear representationsteering. arXiv preprint arXiv:2412.12359, 2024.
- [9] Liwei Che, Tony Qingze Liu, Jing Jia, Weiyi Qin, Ruixiang Tang, and Vladimir Pavlovic. Eazy: Eliminating hallucinations in lvlms by zeroing out hallucinatory image tokens. arXiv preprint arXiv:2503.07772, 2025.
- [10] Dongping Chen, Ruoxi Chen, Shilin Zhang, Yaochen Wang, Yinuo Liu, Huichi Zhou, Qihui Zhang, Yao Wan, Pan Zhou, and Lichao Sun. Mllm-as-a-judge: Assessing multimodal llm-as-ajudge with vision-language benchmark. In ICML, 2024.
- [11] Yukun Chen, Shuo Shao, Enhao Huang, Yiming Li, Pin-Yu Chen, Zhan Qin, and Kui Ren. Refine: Inversion-free backdoor defense via model reprogramming. ICLR, 2025.
- [12] Zhe Chen, Weiyun Wang, Yue Cao, Yangzhou Liu, Zhangwei Gao, Erfei Cui, Jinguo Zhu, Shenglong Ye, Hao Tian, Zhaoyang Liu, et al. Expanding performance boundaries of open-source multimodal models with model, data, and test-time scaling. arXiv preprint arXiv:2412.05271, 2024.
- [13] Wei-Lin Chiang, Zhuohan Li, Zi Lin, Ying Sheng, Zhanghao Wu, Hao Zhang, Lianmin Zheng, Siyuan Zhuang, Yonghao Zhuang, Joseph E. Gonzalez, Ion Stoica, and Eric P. Xing. Vicuna: An open-source chatbot impressing gpt-4 with 90%* chatgpt quality, March 2023.
- [14] Edward Chou, Florian Tramer, and Giancarlo Pellegrino. Sentinet: Detecting localized universal attacks against deep learning systems. In 2020 IEEE Security and Privacy Workshops (SPW), pages 48–54. IEEE, 2020.
- [15] Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, et al. Palm: Scaling language modeling with pathways. JMLR, pages 1–113, 2023.
- [16] Can Cui, Yunsheng Ma, Xu Cao, Wenqian Ye, Yang Zhou, Kaizhao Liang, Jintai Chen, Juanwu Lu, Zichong Yang, Kuei-Da Liao, et al. A survey on multimodal large language models for autonomous driving. In WACV, pages 958–979, 2024.
- [17] Yi Ding, Lijun Li, Bing Cao, and Jing Shao. Rethinking bottlenecks in safety fine-tuning of vision language models. arXiv preprint arXiv:2501.18533, 2025.

- [18] Yinpeng Dong, Huanran Chen, Jiawei Chen, Zhengwei Fang, Xiao Yang, Yichi Zhang, Yu Tian, Hang Su, and Jun Zhu. How robust is google’s bard to adversarial image attacks? arXiv preprint arXiv:2309.11751, 2023.
- [19] Yiyang Fang, Jian Liang, Wenke Huang, He Li, Kehua Su, and Mang Ye. Catch your emotion: Sharpening emotion perception in multimodal large language models. In ICML, 2025.
- [20] Jiahui Gao, Renjie Pi, Tianyang Han, Han Wu, Lanqing Hong, Lingpeng Kong, Xin Jiang, and Zhenguo Li. Coca: Regaining safety-awareness of multimodal large language models with constitutional calibration. arXiv preprint arXiv:2409.11365, 2024.
- [21] Yansong Gao, Yeonjae Kim, Bao Gia Doan, Zhi Zhang, Gongxuan Zhang, Surya Nepal, Damith C Ranasinghe, and Hyoungshick Kim. Design and evaluation of a multi-domain trojan detection method on deep neural networks. TDSC, pages 2349–2364, 2021.
- [22] Yansong Gao, Change Xu, Derui Wang, Shiping Chen, Damith C Ranasinghe, and Surya Nepal. Strip: A defence against trojan attacks on deep neural networks. In ACSAC, pages 113–125, 2019.
- [23] Yichen Gong, Delong Ran, Jinyuan Liu, Conglei Wang, Tianshuo Cong, Anyu Wang, Sisi Duan, and Xiaoyun Wang. Figstep: Jailbreaking large vision-language models via typographic visual prompts. arXiv preprint arXiv:2311.05608, 2023.
- [24] Yunhao Gou, Kai Chen, Zhili Liu, Lanqing Hong, Hang Xu, Zhenguo Li, Dit-Yan Yeung, James T Kwok, and Yu Zhang. Eyes closed, safety on: Protecting multimodal llms via imageto-text transformation. In ECCV, pages 388–404. Springer, 2024.
- [25] Linshan Hou, Ruili Feng, Zhongyun Hua, Wei Luo, Leo Yu Zhang, and Yiming Li. Ibd-psc: Input-level backdoor detection via parameter-oriented scaling consistency. ICML, 2024.
- [26] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al. Lora: Low-rank adaptation of large language models. ICLR, page 3, 2022.
- [27] Kunzhe Huang, Yiming Li, Baoyuan Wu, Zhan Qin, and Kui Ren. Backdoor defense via decoupling the training process. ICLR, 2022.
- [28] Tiansheng Huang, Sihao Hu, Fatih Ilhan, Selim Furkan Tekin, and Ling Liu. Harmful fine-tuning attacks and defenses for large language models: A survey. arXiv preprint arXiv:2409.18169, 2024.
- [29] Tiansheng Huang, Sihao Hu, and Ling Liu. Vaccine: Perturbation-aware alignment for large language models against harmful fine-tuning attack. NeurIPS, 2024.
- [30] Wenke Huang, Jian Liang, Xianda Guo, Yiyang Fang, Guancheng Wan, Xuankun Rong, Chi Wen, Zekun Shi, Qingyun Li, Didi Zhu, et al. Keeping yourself is important in downstream tuning multimodal large language model. arXiv preprint arXiv:2503.04543, 2025.
- [31] Wenke Huang, Jian Liang, Zekun Shi, Didi Zhu, Guancheng Wan, He Li, Bo Du, Dacheng Tao, and Mang Ye. Learn from downstream and be yourself in multimodal large language model fine-tuning. arXiv preprint arXiv:2411.10928, 2024.
- [32] Mojan Javaheripi, Mohammad Samragh, Gregory Fields, Tara Javidi, and Farinaz Koushanfar. Cleann: Accelerated trojan shield for embedded neural networks. In ICCD, pages 1–9, 2020.
- [33] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In International conference on machine learning, pages 19730–19742. PMLR, 2023.
- [34] Yifan Li, Hangyu Guo, Kun Zhou, Wayne Xin Zhao, and Ji-Rong Wen. Images are achilles’ heel of alignment: Exploiting visual vulnerabilities for jailbreaking multimodal large language models. arXiv preprint arXiv:2403.09792, 2024.
- [35] Yige Li, Xixiang Lyu, Nodens Koren, Lingjuan Lyu, Bo Li, and Xingjun Ma. Neural attention distillation: Erasing backdoor triggers from deep neural networks. ICLR, 2021.

- [36] Yiming Li, Yong Jiang, Zhifeng Li, and Shu-Tao Xia. Backdoor learning: A survey. IEEE TNNLS, pages 5–22, 2022.
- [37] Yinshan Li, Hua Ma, Zhi Zhang, Yansong Gao, Alsharif Abuadbba, Anmin Fu, Yifeng Zheng, Said F Al-Sarawi, and Derek Abbott. Ntd: Non-transferability enabled backdoor detection. IEEE TIFS, 2021.
- [38] Yuetai Li, Zhangchen Xu, Fengqing Jiang, Luyao Niu, Dinuka Sahabandu, Bhaskar Ramasubramanian, and Radha Poovendran. Cleangen: Mitigating backdoor attacks for generation tasks in large language models. EMNLP, 2024.
- [39] Jian Liang, Wenke Huang, Guancheng Wan, Qu Yang, and Mang Ye. Lorasculpt: Sculpting lora for harmonizing general and specialized knowledge in multimodal large language models. CVPR, 2025.
- [40] Jiawei Liang, Siyuan Liang, Man Luo, Aishan Liu, Dongchen Han, Ee-Chien Chang, and Xiaochun Cao. Vl-trojan: Multimodal instruction backdoor attacks against autoregressive visual language models. arXiv preprint arXiv:2402.13851, 2024.
- [41] Siyuan Liang, Jiawei Liang, Tianyu Pang, Chao Du, Aishan Liu, Ee-Chien Chang, and Xiaochun Cao. Revisiting backdoor attacks against large vision-language models. arXiv preprint arXiv:2406.18844, 2024.
- [42] Aristidis Likas, Nikos Vlassis, and Jakob J Verbeek. The global k-means clustering algorithm. PR, pages 451–461, 2003.
- [43] Ji Lin, Hongxu Yin, Wei Ping, Pavlo Molchanov, Mohammad Shoeybi, and Song Han. Vila: On pre-training for visual language models. In CVPR, pages 26689–26699, 2024.
- [44] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. NeurIPS, 36:34892–34916, 2023.
- [45] Yue Liu, Shengfang Zhai, Mingzhe Du, Yulin Chen, Tri Cao, Hongcheng Gao, Cheng Wang, Xinfeng Li, Kun Wang, Junfeng Fang, Jiaheng Zhang, and Bryan Hooi. Guardreasoner-vl: Safeguarding vlms via reinforced reasoning. arXiv preprint arXiv:2505.11049, 2025.
- [46] Yuntao Liu, Ankit Mondal, Abhishek Chakraborty, Michael Zuzak, Nina Jacobsen, Daniel Xing, and Ankur Srivastava. Neural trojans. In ESCP, pages 1648–1655. Springer, 2025.
- [47] Sylvain Lobry, Diego Marcos, Jesse Murray, and Devis Tuia. Rsvqa: Visual question answering for remote sensing data. IEEE TGRS, pages 8555–8566, 2020.
- [48] Pan Lu, Swaroop Mishra, Tanglin Xia, Liang Qiu, Kai-Wei Chang, Song-Chun Zhu, Oyvind Tafjord, Peter Clark, and Ashwin Kalyan. Learn to explain: Multimodal reasoning via thought chains for science question answering. NeurIPS, 35:2507–2521, 2022.
- [49] Pan Lu, Liang Qiu, Jiaqi Chen, Tony Xia, Yizhou Zhao, Wei Zhang, Zhou Yu, Xiaodan Liang, and Song-Chun Zhu. Iconqa: A new benchmark for abstract diagram understanding and visual language reasoning. arXiv preprint arXiv:2110.13214, 2021.
- [50] Weimin Lyu, Lu Pang, Tengfei Ma, Haibin Ling, and Chao Chen. Trojvlm: Backdoor attack against vision language models. arXiv preprint arXiv:2409.19232, 2024.
- [51] Weimin Lyu, Jiachen Yao, Saumya Gupta, Lu Pang, Tao Sun, Lingjie Yi, Lijie Hu, Haibin Ling, and Chao Chen. Backdooring vision-language models with out-of-distribution data. arXiv preprint arXiv:2410.01264, 2024.
- [52] Siyuan Ma, Weidi Luo, Yu Wang, Xiaogeng Liu, Muhao Chen, Bo Li, and Chaowei Xiao. Visualroleplay: Universal jailbreak attack on multimodal large language models via role-playing image characte. arXiv preprint arXiv:2405.20773, 2024.
- [53] Zhenyang Ni, Rui Ye, Yuxi Wei, Zhen Xiang, Yanfeng Wang, and Siheng Chen. Physical backdoor attack can jeopardize driving with vision-large-language models. arXiv preprint arXiv:2404.12916, 2024.

- [54] OpenAI. Fine-tuning. https://platform.openai.com/docs/guides/fine-tuning.
- [55] Renjie Pi, Tianyang Han, Jianshu Zhang, Yueqi Xie, Rui Pan, Qing Lian, Hanze Dong, Jipeng Zhang, and Tong Zhang. Mllm-protector: Ensuring mllm’s safety without hurting performance. arXiv preprint arXiv:2401.02906, 2024.
- [56] Renjie Pi, Jianshu Zhang, Tianyang Han, Jipeng Zhang, Rui Pan, and Tong Zhang. Personalized visual instruction tuning. ICLR, 2024.
- [57] Renjie Pi, Jianshu Zhang, Jipeng Zhang, Rui Pan, Zhekai Chen, and Tong Zhang. Image textualization: An automatic framework for creating accurate and detailed image descriptions. NeurIPS, 2024.
- [58] Fanchao Qi, Yangyi Chen, Mukai Li, Yuan Yao, Zhiyuan Liu, and Maosong Sun. Onion: A simple and effective defense against textual backdoor attacks. EMNLP, 2021.
- [59] Xiangyu Qi, Kaixuan Huang, Ashwinee Panda, Peter Henderson, Mengdi Wang, and Prateek Mittal. Visual adversarial examples jailbreak aligned large language models. In AAAI, pages 21527–21536, 2024.
- [60] Xiangyu Qi, Yi Zeng, Tinghao Xie, Pin-Yu Chen, Ruoxi Jia, Prateek Mittal, and Peter Henderson. Fine-tuning aligned language models compromises safety, even when users do not intend to! arXiv preprint arXiv:2310.03693, 2023.
- [61] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In ICML, pages 8748–8763. PMLR, 2021.
- [62] Douglas A Reynolds et al. Gaussian mixture models. Encyclopedia of biometrics, page 3, 2009.
- [63] Christian Schlarmann and Matthias Hein. On the adversarial robustness of multi-modal foundation models. In ICCV, pages 3677–3685, 2023.
- [64] Yucheng Shi, Mengnan Du, Xuansheng Wu, Zihan Guan, Jin Sun, and Ninghao Liu. Black-box backdoor defense via zero-shot image purification. NeurIPS, 36:57336–57366, 2023.
- [65] Xiaofei Sun, Xiaoya Li, Yuxian Meng, Xiang Ao, Lingjuan Lyu, Jiwei Li, and Tianwei Zhang. Defending against backdoor attacks in natural language generation. In AAAI, pages 5257–5265, 2023.
- [66] Ruixiang Tang, Jiayi Yuan, Yiming Li, Zirui Liu, Rui Chen, and Xia Hu. Setting the trap: Capturing and defeating backdoor threats in plms through honeypots. NeurIPS, 2023.
- [67] Gemini Team, Rohan Anil, Sebastian Borgeaud, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, Katie Millican, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023.
- [68] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023.
- [69] Minh-Hao Van, Prateek Verma, and Xintao Wu. On large visual language models for medical imaging analysis: An empirical study. In CHASE, pages 172–176. IEEE, 2024.
- [70] Ramakrishna Vedantam, C Lawrence Zitnick, and Devi Parikh. Cider: Consensus-based image description evaluation. In CVPR, pages 4566–4575, 2015.
- [71] Sudong Wang, Yunjian Zhang, Yao Zhu, Jianing Li, Zizhe Wang, Yanwei Liu, and Xiangyang Ji. Towards understanding how knowledge evolves in large vision-language models. arXiv preprint arXiv:2504.02862, 2025.
- [72] Yibo Wang, Tiansheng Huang, Li Shen, Huanjin Yao, Haotian Luo, Rui Liu, Naiqiang Tan, Jiaxing Huang, and Dacheng Tao. Panacea: Mitigating harmful fine-tuning for large language models via post-fine-tuning perturbation. arXiv preprint arXiv:2501.18100, 2025.

- [73] Yu Wang, Xiaogeng Liu, Yu Li, Muhao Chen, and Chaowei Xiao. Adashield: Safeguarding multimodal large language models from structure-based attack via adaptive shield prompting. ECCV, 2024.
- [74] Yu Wang, Xiaofei Zhou, Yichen Wang, Geyuan Zhang, and Tianxing He. Jailbreak large visual language models through multi-modal linkage. arXiv preprint arXiv:2412.00473, 2024.
- [75] Shicheng Xu, Liang Pang, Yunchang Zhu, Huawei Shen, and Xueqi Cheng. Cross-modal safety mechanism transfer in large vision-language models. arXiv preprint arXiv:2410.12662, 2024.
- [76] Xiong Xu, Kunzhe Huang, Yiming Li, Zhan Qin, and Kui Ren. Towards reliable and efficient backdoor trigger inversion via decoupling benign features. In ICLR, 2023.
- [77] Yue Xu, Xiuyuan Qi, Zhan Qin, and Wenjie Wang. Cross-modality information check for detecting jailbreaking in multimodal large language models. EMNLP, 2024.
- [78] Jinluan Yang, Anke Tang, Didi Zhu, Zhengyu Chen, Li Shen, and Fei Wu. Mitigating the backdoor effect for multi-task model merging via safety-aware subspace. ICLR, 2024.
- [79] Mang Ye, Xuankun Rong, Wenke Huang, Bo Du, Nenghai Yu, and Dacheng Tao. A survey of safety on large vision-language models: Attacks, defenses and evaluations. arXiv preprint arXiv:2502.14881, 2025.
- [80] Shukang Yin, Chaoyou Fu, Sirui Zhao, Ke Li, Xing Sun, Tong Xu, and Enhong Chen. A survey on multimodal large language models. arXiv preprint arXiv:2306.13549, 2023.
- [81] Peter Young, Alice Lai, Micah Hodosh, and Julia Hockenmaier. From image descriptions to visual denotations: New similarity metrics for semantic inference over event descriptions. TACL, 2:67–78, 2014.
- [82] Zenghui Yuan, Jiawen Shi, Pan Zhou, Neil Zhenqiang Gong, and Lichao Sun. Badtoken: Tokenlevel backdoor attacks to multi-modal large language models. arXiv preprint arXiv:2503.16023, 2025.
- [83] Jianshu Zhang, Dongyu Yao, Renjie Pi, Paul Pu Liang, and Yi R Fung. Vlm2-bench: A closer look at how well vlms implicitly link explicit matching visual cues. arXiv preprint arXiv:2502.12084, 2025.
- [84] Jiarui Zhang, Mahyar Khayatkhoei, Prateek Chhikara, and Filip Ilievski. Mllms know where to look: Training-free perception of small visual details with multimodal llms. ICLR, 2025.
- [85] Xiaofeng Zhang, Yihao Quan, Chen Shen, Xiaosong Yuan, Shaotian Yan, Liang Xie, Wenxiao Wang, Chaochen Gu, Hao Tang, and Jieping Ye. From redundancy to relevance: Information flow in lvlms across reasoning tasks. NAACL, 2024.
- [86] Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. Minigpt-4: Enhancing vision-language understanding with advanced large language models. arXiv preprint arXiv:2304.10592, 2023.
- [87] Yongshuo Zong, Ondrej Bohdal, Tingyang Yu, Yongxin Yang, and Timothy Hospedales. Safety fine-tuning at (almost) no cost: A baseline for vision large language models. ICML, 2024.

