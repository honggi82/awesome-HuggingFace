## Robust-U1: Can MLLMs Self-Recover Corrupted Visual Content for Robust Understanding?

Jiaqi Tang1* Jianmin Chen2* Youyang Zhai2* Wei Wei2 Runtao Liu1 Mengjie Zhao3 Xiangyu Wu4 Qingfa Xiao5 Qifeng Chen1

# arXiv:2606.08063v1[cs.CV]6Jun2026

### Abstract

Multimodal Large Language Models (MLLMs) have demonstrated remarkable success in visual understanding, yet their performance degrades significantly under real-world visual corruptions. While existing robustness enhancement approaches exist, they are limited: black-box feature alignment lacks interpretability, and whitebox text-based reasoning cannot restore lost pixellevel details. This work investigates a fundamental research question: Can MLLMs recover corrupted visual content by themselves? To address this, we propose Robust-U1, a novel framework that equips MLLMs with explicit visual self-recovery capability for robust understanding. The approach comprises three core stages: supervised fine-tuning for initial reconstruction, reinforcement learning with dual rewards (pixellevel SSIM and semantic-level CLIP similarity) for aligning high visual quality, and multimodal reasoning that jointly considers both the corrupted input and the recovered image. Extensive experiments demonstrate that Robust-U1 achieves state-of-the-art robustness on the real-world corruption benchmark and maintains superior performance under adversarial corruptions on general VQA benchmarks. Analysis confirms that high-quality visual recovery directly enhances reasoning performance, establishing self-recovery as a critical mechanism for robust visual understanding. The source code is available at https: //github.com/jqtangust/Robust-U1.

*Equal contribution 1The Hong Kong University of Science and Technology, Hong Kong 2Northwestern Polytechnical University, Xi’an, China 3Northeastern University, Shenyang, China 4Nanjing University of Science and Technology, Nanjing, China 5The Hong Kong University of Science and Technology (Guangzhou), Guangzhou, China. Correspondence to: Qifeng Chen <cqf@ust.hk>, Wei Wei <weiweinwpu@nwpu.edu.cn>.

Proceedings of the 43rd International Conference on Machine Learning, Seoul, South Korea. PMLR 306, 2026. Copyright 2026 by the author(s).

### 1. Introduction

Multimodal Large Language Models (MLLMs) have achieved unprecedented performance in visual understanding by effectively aligning visual and textual representations through large-scale pretraining (Liu et al., 2024; Bai

- et al., 2025), enabling diverse downstream applications such as open-world video anomaly understanding (Tang et al., 2024a). However, their practical deployment is critically hindered by a pronounced vulnerability to real-world visual corruptions (or degradations), such as system noise (Tang
- et al., 2026a), compression artifacts (Liu et al., 2025a), and adverse weather effects (Lai et al., 2025). These corruptions severely disrupt the visual features, leading to a dramatic decrease in performance across various downstream tasks (Tang et al., 2026a; Qiu et al., 2025).

To mitigate the above issue, current robustness enhancement methods predominantly rely on black-box strategies that align features of corrupted and clean images within the visual encoder via adversarial alignment (Mao et al., 2023; Hossain & Imteaj, 2024; Schlarmann et al., 2024) (as shown in Fig. 1-(A)). While improving performance, these methods lack interpretability and fail to explicitly model the corruption process. In contrast, recent work introduces a white-box paradigm that employs an explicit textual reasoning chain to describe corruption types and their semantic impacts (Tang et al., 2026a), thereby enhancing interpretability (as described in Fig. 1-(B)).

However, existing methods remain constrained by their reliance on the textual description (Tang et al., 2026a), which cannot represent the pixel-level details for faithful visual understanding (as shown in Fig. 1-(C)). This limitation motivates a pivotal research direction: Can MLLMs Recover Corrupted Visual Content by Themselves? If so, it would establish a more intrinsic robustness for understanding corrupted images. Achieving such self-recovery capability would provide a more complete solution to corruption robustness, where the model actively restores lost information rather than merely offering text-based compensation.

To address these limitations, we propose Robust-U1, a novel framework that equips MLLMs with explicit visual

###### (A) Alignment

(B) Text-based Reasoning

###### (C) Self-Recovering

|[Figure 1]|
|---|

|[Figure 2]|
|---|

What is next to each pizza pie?

|[Figure 3]|
|---|

|[Figure 4]|
|---|

|[Figure 5]|
|---|

|[Figure 6]|
|---|

What is next to

What is next to

each

each

Clear Corrupted

###### Clear Corrupted

###### Clear Corrupted

pizza pie?

pizza pie?

###### Questions

###### Questions

###### Questions

Visual Tokenizer

Visual Tokenizer

Visual Tokenizer

Feature

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

| | |
|---|---|
| | |

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

Alignment

Multimodal Large Language Model

Multimodal Large Language Model

Multimodal Large Language Model

|[Figure 7]|
|---|

###### Recovering…

###### Thinking…

knife

Text-based CoT -> knife

Multimodal CoT -> knife

Black-Box

Lack of

Text-based

No Visual

Visual

Multimodal

Setting

Interpretability

Reasoning

Representation

Recovery

Reasoning

- Figure 1. Comparison of robustness enhancement paradigms. (A) Implicit Adaptation: Black-box feature alignment within the visual encoder. (B) Text-based Reasoning: White-box textual chain describing corruption impacts. (C) Our Robust-U1 (Self-Recovering): Direct visual self-recovery and multimodal reasoning over both corrupted and recovered images.

recovery capability, enabling direct recovery of corrupted images and leveraging these restored visuals to enhance robust visual understanding. Our methodology comprises three stages: First, we conduct supervised fine-tuning on the large-scale real-world image reconstruction dataset to endow the MLLM with foundational reconstruction ability

bustness that moves beyond implicit feature alignment or text-only reasoning.

- • We design a three-stage training pipeline: (i) supervised fine-tuning to acquire foundational visual recovery ability, (ii) reinforcement learning with pixel-semantic rewards (SSIM and CLIP similarity) to ensure highfidelity reconstruction, and (iii) multimodal reasoning that jointly leverages both corrupted and recovered content for robust visual understanding.
- • Extensive experiments show Robust-U1 achieves SOTA robustness across real-world and adversarial corruption benchmarks. Our analysis further confirms that the high-quality visual recovery directly enhances reasoning performance, validating self-recovery as a critical mechanism for robust visual understanding.

- (Section 3.1). Second, we employ reinforcement learning with dual rewards including (i) Pixel-Level Structural Reward (supported by SSIM) and (ii) Semantic Consistency Reward (built on CLIP (Wu et al., 2023)), to align reconstructions with both structural and semantic fidelity
- (Section 3.2). Finally, the model learns to integrate both corrupted and recovered visual content to perform multimodal reasoning, synthesizing a more robust visual understanding
- (Section 3.3).

Comprehensive experiments validate our approach across multiple dimensions. On the real-world corruption benchmark R-Bench (Li et al., 2024), Robust-U1 significantly outperforms existing robust MLLMs across all corruption intensities. When subjected to adversarial corruptions on general VQA benchmarks including MMMB (Sun et al., 2025), MMStar (Chen et al., 2024a), and RealWorldQA (xAI, 2024), Robust-U1 maintains superior robustness with minimal performance decrease. Furthermore, our extensive analyses demonstrate that the recovered images achieve high quality and that this visual reconstruction directly contributes to improved reasoning performance, thereby confirming the critical role of self-recovery in achieving robust visual understanding. Our contributions are threefold:

### 2. Related Works

(B) Ours: Explicitly Reasoning (Robust-R1)

Corruption Robustness of MLLMs Multimodal large language models (MLLMs) remain vulnerable to environmental perturbations (Liu et al., 2025a; Hu et al., 2026), which frequently impair their visual perception capabilities and lead to significant performance drops (Li et al., 2024). This has made robustness enhancement a central objective in visual understanding research. Prevailing methods can be primarily divided into two categories: implicit alignment and text-based reasoning. The former, exemplified by works like TeCoA (Mao et al., 2023), Robust LLaVA (Malik et al., 2025), and Robust CLIP (Schlarmann et al., 2024), employs adversarial fine-tuning of the visual encoder to resist localized distortions (Fig. 1-A). However, their dependence on limited adversarial datasets often compromises generalization. In contrast, the latter paradigm, as seen in Robust-R1 (Tang et al., 2026a), enhances interpretability by

• We propose Robust-U1, a novel framework that for the first time empowers MLLMs with explicit visual self-recovery capability. This enables pixel-level visual reconstruction of corrupted content for corruption ro-

Stage III: Multimodal Reasoning for

###### Stage I: SFT for Visual Self-Recovery

Stage II: Enhancing Visual Quality via RL

Robust Visual Understanding

###### Flow-GRPO

###### Rectified-Flow Loss

Here is the restored image：

Next-Token Prediction

Pixel-Level Structural Reward + Semantic Consistency Reward

<restored image>.

<THINKING> … the presence of a person being moved by visible apparatus … the presence of large kite-like structures confirms the use of wind sails.

|[Figure 8]|
|---|

|[Figure 9]|
|---|

|[Figure 10]|
|---|

|[Figure 11]|
|---|

|[Figure 12]|
|---|

Restored Images

Restored

Restored

Ground

Ground

Images

Images

Truth

Truth

<THINKING_END>

<ANSWER>wind sails<ANSWER_END>

Und. Expert Gen. Expert

Und. Expert Gen. Expert

Und. Expert Gen. Expert

Unified Multimodal Large Language Model

Unified Multimodal Large Language Model

Unified Multimodal Large Language Model

Text Tokenizer

Text Tokenizer

Text Tokenizer

Und.

Gen.

Und.

Gen.

Und. Encoder

Gen. Encoder

Encoder

Encoder

Encoder

Encoder

|[Figure 13]|
|---|

|[Figure 14]|
|---|

|[Figure 15]|
|---|

What propels

Corrupted

Corrupted

Corrupted

these people across the water?

Questions

Questions

Questions

N/A

N/A

Images

Images

Images

- Figure 2. Overview of the three-stage Robust-U1 framework. Stage I: Supervised Fine-Tuning adapts the unified MLLM to recover clean images from corrupted inputs using a rectified-flow loss. Stage II: Reinforcement Learning with dual rewards further enhances the quality of the recovered images via Flow-GRPO (Liu et al., 2025b). Stage III: Multimodal Reasoning trains the model to answer questions by jointly analyzing both the corrupted and the recovered images, leading to robust understanding.

incorporating explicit textual chains that describe corruption types and their semantic consequences (Fig. 1-B).

However, these methods remain constrained to the textual modality and cannot restore the lost visual information. Our framework pioneers a novel framework that explicitly recovers corrupted visual content through self-recovery, as shown in Fig. 1-(C). By directly reconstructing the corrupted image and leveraging reasoning, Robust-U1 provides enhanced robustness for accurate understanding.

Think with Images Early reasoning approaches focused on generating text-only chains of thought to guide inference, relying solely on the linguistic modality for intermediate reasoning steps (Wei et al., 2022; Liu et al., 2024). Recent work (Zheng et al., 2025) has sought to incorporate visual representations into the reasoning process, which enhances reasoning by recalling or generating intermediate visual features or descriptions, with multi-agent extensions further coordinating specialized agents for long-form video reasoning (Liu et al., 2026). However, these methods primarily operate on the visual information already present or assumed within the scene. More recent explorations, such as Thinking with Generated Images (Chern et al., 2025), have begun to investigate generating auxiliary visual representations, either via internal model or external tools, to augment reasoning further.

Our method inherits and extends previous paradigms. We equip MLLMs with self-recovery capabilities, enabling them to generate the recovered corrupted image to facilitate robust visual understanding.

- 3. Methodology

and a textual query Q are processed by a Multimodal Large Language Model FMLLM to produce an answer Ao:

Ao = FMLLM(Io, Q; Θ), (1) where Θ denotes the model parameters. However, in realworld scenarios, images are often corrupted by various degradations, which can be modeled as Ic = D(Io), where D denotes the corruption function. The performance of standard MLLMs degrades significantly when processing such corrupted inputs (Li et al., 2024; Tang et al., 2026a; Long et al., 2025).

Our Insight To address this limitation, we propose a robust framework that explicitly incorporates a visual selfrecovery process. Our approach first reconstructs a recovered image Ir through an approximate inverse of the corruption process. The complete formulation of our robust model can be expressed as:

A = FMLLM(Robust)(Ic, Q; Θ) = FMLLM D−1(Ic)

, Ic, Q; Θ , (2)

Ir

where D−1 : Ic  → Ir represents the self-recovery module that approximates the inverse of the corruption process,

and FMLLM(Robust) denotes our robust multimodal reasoning function that synthesizes information from both the original

corrupted input Ic and the recovered image Ir to generate the final answer A. This formulation enables the model to actively compensate for information loss while maintaining awareness of the original corruption characteristics.

Overview Based on the above formulation, we propose a three-stage framework, Robust-U1, to realize robust visual understanding through explicit self-recovery and multimodal reasoning. First, we perform supervised finetuning to adapt a pre-trained Unified MLLM for visual selfrecovery, enabling it to reconstruct a recovered image Ir

Problem Formulation Consider a standard multimodal understanding pipeline where a clean image Io ∈ RH×W×3

from the corrupted input Ic, as shown in Fig. 2-(Stage I). Second, we employ reinforcement learning with dual rewards in both pixel-level and semantic-level for semantic consistency, to further align the recovered image Ir with high quality, as shown in Fig. 2-(Stage II).. Finally, we train the model to perform multimodal reasoning by jointly considering both the corrupted image Ic and the recovered image Ir, thereby synthesizing a robust visual understanding, as shown in Fig. 2-(Stage III).

#### 3.1. Supervised Fine-Tuning for Visual Self-Recovery

To endow the MLLM with the ability to recover clean images from corrupted inputs, we build our base model on a pre-trained unified MLLM, BAGEL (Deng et al., 2025), that inherently supports both multimodal understanding and generation. Our goal is to specialize this general generative capability into a dedicated self-recovery module D−1, inspired by recent advances in learning-based visual recovery from physical degradations (Tang et al., 2024c).

The recovery process is implemented by conditioning the unified MLLM on a specific recovery prompt Prec (e.g., “Recover the clean version of this corrupted image.”). We adopt the rectified flow formulation (Liu et al., 2023) used in state-of-the-art generative unified models. Specifically, the image is first encoded into a latent representation Zc. The model is then trained to denoise a noisy version of the clean latent Zo conditioned on Zc and Prec. The objective function is defined as:

LSFT = Et∼U(0,1),ϵ∼N(0,I) ∥ϵ − ϵΘ(Zc, Zo(t), t, Prec)∥2 ,

(3)

where Zo(t) = (1−t)Zo +tϵ is the noisy latent at timestep t, and ϵΘ is the noise prediction network parameterized by the model weights Θ. This objective directly optimizes the model to reverse the corruption process in the latent space, effectively learning the inverse mapping D−1.

After training, the recovered image Ir is obtained by decoding the denoised latent representation. This supervised fine-tuning stage transforms the model’s general image generation capability into a specialized visual self-recovery module, providing the essential first component of our robust understanding pipeline.

#### 3.2. Aligning Higher Visual Quality through Reinforcement Learning

While supervised fine-tuning establishes a foundational recovery capability, the recovered images may still lack precise alignment with the original clean images in terms of both structural and semantic fidelity. To further enhance the quality of the recovered image Ir, we employ reinforcement learning (RL) with a dual-reward objective, optimizing the self-recovery module D−1 using the Flow-GRPO (Liu et al.,

(B) Semantic Consistency Reward

(A) Pixel-Level Structural Reward

Ground Restored Truth

###### Ground Restored Truth

| | | | | | |
|---|---|---|---|---|---|
| |[Figure 16]| |······| | |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
|[Figure 17]| | |······| | |
| | | | | | |

|[Figure 18]|
|---|

|[Figure 19]|
|---|

Visual Encoder

Visual Encoder (CLIP)

Local Windows

(CLIP)

| | | | | | | |
|---|---|---|---|---|---|---|

| | | | | | | |
|---|---|---|---|---|---|---|

[Figure 20]

Ground Truth Embedding

Restored Embedding

Y

Luminance Contrast Structure

|𝐜𝐨𝐬 𝛉|
|---|

Cosine

|𝐑𝐬𝐞𝐦 =|
|---|

Similarity

𝐑𝐩𝐢𝐱 = 𝒍 + 𝑪 + 𝑺

0 X

Figure 3. Schematic of the dual-reward mechanism used in the reinforcement learning stage. (A) Pixel-Level Structural Reward: Computes the SSIM index by comparing local patches (luminance, contrast, structure) between the recovered image Ir and the ground-truth clean image Io. (B) Semantic Consistency Reward: Utilizes a frozen TinyCLIP (Wu et al., 2023) model to extract image embeddings. The reward is derived from the cosine similarity between the embeddings of Ir and Io, encouraging semantic alignment in the vision-language feature space.

2025b). This approach allows us to directly optimize for both pixel-level accuracy and semantic consistency, which are crucial for downstream visual understanding tasks.

Pixel-Level Structural Reward To ensure the recovered image Ir is structurally similar to the ground-truth clean image Io, we employ the Structural Similarity Index Measure (SSIM) (Hore & Ziou, 2010) as a pixel-level reward in Fig. 3-(A). SSIM is computed over local image patches and combines three independent comparisons: luminance (l), contrast (c), and structure (s). The reward is defined as:

Rpix(Ir, Io) = SSIM(Ir, Io)

N

(4)

1 N

l(pir, pio) · c(pir, pio) · s(pir, pio) ,

=

i=1

where N is the number of local patches, and pir,pio denote the i-th patch from Ir and Io, respectively. The three components are given by:

2µrµo + C1 µ2r + µ2o + C1

, (5)

l(pr,po) =

2σrσo + C2 σr2 + σo2 + C2

, (6)

c(pr,po) =

σro + C3 σrσo + C3

, (7)

s(pr,po) =

with µr,µo the patch means, σr,σo the patch standard deviations, and σro their covariance. The constants C1,C2,C3 are small values added for numerical stability. The final score lies in [0,1] in paired images, where higher values indicate better structural preservation.

Semantic Consistency Reward Pixel-level similarity alone may not guarantee semantic correctness. To ensure

the recovered image preserves the semantic content of the original, we introduce a semantic consistency reward using a frozen CLIP model MCLIP (Wu et al., 2023) in Fig. 3-(B). First, we compute the cosine similarity between the CLIP embeddings of the recovered image Ir and the original clean image Io:

Sim (MCLIP(Ir), MCLIP(Io)) = MCLIP(Ir) · MCLIP(Io)

. (8)

∥MCLIP(Ir)∥∥MCLIP(Io)∥

Then, we transform this similarity into a reward signal that penalizes semantic deviation. The reward is defined as:

of the ground-truth (with a reasoning chain):

L

log PΘ(a∗t | a∗<t, Ic, Ir, Q),

LMLLM = −E(Ic,Ir,Q,A∗)

t=1

(10)

where a∗t denotes the t-th token of the target answer A∗. This enables the model to learn how to integrate information from both views for robust understanding.

Through this stage, the model learns to leverage the recovered image for primary content understanding while consulting the corrupted image to resolve ambiguities, leading to more reliable performance under corruption.

Rsem(Ir, Io) = exp (−α · (1 − Sim (MCLIP(Ir), MCLIP(Io)))) ,

(9)

where α > 0 is a scaling factor that controls the sensitivity of the reward to semantic deviations. This formulation ensures that the reward is maximized (equal to 1) when the similarity is 1, and decays exponentially as the similarity decreases, thereby encouraging the model to generate recoveries that are semantically consistent with the original image in the joint vision-language embedding space.

Optimization We optimize the self-recovery module using Flow-GRPO (Liu et al., 2025b), which formulates the denoising process as a Markov Decision Process and employs group-based relative policy optimization, akin to recent group-relative preference optimization for visual interaction tasks (Tang et al., 2026b). For each corrupted image Ic, we sample a group of G trajectories using stochastic sampling (enabled by converting the deterministic ODE to an SDE), resulting in recovered images {Iir}Gi=1. The advantage for each trajectory is computed via group normalization of the composite rewards. The policy is then updated to maximize the expected advantage while constrained by a KL divergence penalty to prevent reward hacking and maintain generation quality.

Through this reinforcement learning stage, the self-recovery module is further aligned with both structural and semantic fidelity, producing recovered images Ir that are both visually and semantically close to the original clean images.

#### 3.3. Multimodal Reasoning for Robust Understanding

After obtaining the recovered image Ir, we train the model to perform robust visual understanding by jointly reasoning over both the corrupted image Ic and the recovered image Ir. This approach allows the model to verify the robustness in visual understanding.

We structure the input as an interleaved sequence of the two images followed by the textual query Q. The model is trained to generate the answer A conditioned on this input. The training objective LMLLM is to maximize the likelihood

### 4. Experiment

Configuration of Training To support the proposed selfrecovery and multimodal reasoning framework, we require a unified model capable of both multimodal understanding and generation. We select BAGEL (Deng et al., 2025) as our base model due to its unified architecture for multimodal understanding and generation. We then adopt a three-stage training strategy: First, we perform supervised fine-tuning (SFT) on ImageNet-C (Xie et al., 2020) for large-scale image reconstruction, establishing a base model with foundational recovery capabilities. Second, we employ Flow-GRPO (Liu et al., 2025b) on the training data from Robust-R1 (Tang et al., 2026a) to align the model with both semantic and pixel-level rewards, refining the recovery quality. Third, we leverage the reasoning chain data from Robust-R1 (Tang et al., 2026a) to train the model to perform multimodal understanding using both the corrupted and recovered images.

Benchmarks Following the evaluation protocol of RobustR1 (Tang et al., 2026a), we also adopt a two-fold evaluation strategy, consistent with established practices in robustness research. First, we assess real-world corruption robustness using R-Bench (Li et al., 2024), a benchmark specifically designed to measure visual understanding under realistic degradations of varying intensities. Second, we evaluate adversarial corruption robustness by synthetically applying multi-type, multi-level real-world degradations to the images of standard visual question answering benchmarks, including MMMB (Sun et al., 2025), MMStar (Chen et al., 2024a), and RealWorldQA (xAI, 2024). This comprehensive protocol allows us to measure both the model’s intrinsic ability to comprehend degraded content and its capacity to maintain performance under challenging visual corruptions.

Baselines We compare our method against state-of-the-art models from two primary categories to ensure a fair and comprehensive evaluation. The first category comprises general MLLMs that are not specifically optimized for robustness, including Qwen2.5-VL-3B (Bai et al., 2025),

Table 1. Quantitative evaluation on R-Bench (Li et al., 2024) for MCQ, VQA, and CAP tasks under three degradation levels (low to high). The best/second best results are shown in Red/Blue, respectively.

MCQ VQA CAP

Category Method

Overall low mid high low mid high low mid high

Qwen2.5-VL-3B (Bai et al., 2025) 0.6411 0.6022 0.5732 0.4872 0.4854 0.4904 0.3778 0.3704 0.3330 0.4845 Gemma3-4B (Team et al., 2025) 0.5823 0.5776 0.5060 0.4865 0.4630 0.4419 0.4048 0.3746 0.3480 0.4649 InternVL-4B (Chen et al., 2024b) 0.6235 0.6024 0.5914 0.4982 0.4539 0.5108 0.3667 0.3041 0.2851 0.4706 BAGEL (Deng et al., 2025) 0.7176 0.6584 0.5793 0.6497 0.6127 0.6150 0.4685 0.4633 0.4288 0.5770

General MLLM

TeCoA (Mao et al., 2023) 0.4647 0.4223 0.4024 0.4687 0.3994 0.4461 0.2111 0.2195 0.1937 0.3586 Robust CLIP (Schlarmann et al., 2024) 0.4705 0.4658 0.4024 0.4503 0.4339 0.4743 0.2290 0.2219 0.1983 0.3718 Robust LLaVA (Malik et al., 2025) 0.3352 0.2608 0.3048 0.2607 0.2212 0.2443 0.0068 0.0065 0.0067 0.1830 Robust-R1 (Tang et al., 2026a) 0.6529 0.6391 0.6097 0.4914 0.4909 0.4980 0.4068 0.3781 0.3484 0.5017

Robust MLLM

Ours Robust-U1 0.7353 0.7329 0.6768 0.7067 0.7164 0.6934 0.8272 0.8059 0.7640 0.7398

Gemma3-4B (Team et al., 2025), InternVL-4B (Chen et al.,

- 2024b), and BAGEL (Deng et al., 2025). The second category consists of robust MLLMs that incorporate explicit robustness enhancements, such as TeCoA (Mao et al., 2023), Robust CLIP (Schlarmann et al., 2024), and Robust LLaVA (Malik et al., 2025). This comparison allows us to situate our method within the broader landscape of both general-purpose and robustness-specialized multimodal models.

#### 4.1. Performance on Real-World Corruptions

We evaluate Robust-U1 on R-Bench (Li et al., 2024), a comprehensive benchmark for real-world corruption robustness. It assesses three task types, Multiple-Choice Questions (MCQ), Visual Question Answering (VQA), and Image Captioning (CAP), across three increasing degradation intensity levels.

Main Results. As shown in Table 1, Robust-U1 achieves state-of-the-art performance across all tasks and intensities. It significantly outperforms both general-purpose MLLMs (e.g., BAGEL (Deng et al., 2025)) and specialized robust MLLMs (e.g., Robust-R1 (Tang et al., 2026a)) on the overall score. Notably, the performance advantage of Robust-U1 becomes more pronounced as corruption severity increases, demonstrating its robustness under challenging conditions.

Case Study. Table 2 presents a concrete example illustrating the failure modes of prior methods and the success of Robust-U1. Qwen2.5-VL (Bai et al., 2025) and RobustR1 (Bai et al., 2025) are misled by the degradation, inferring an incorrect answer. BAGEL (Deng et al., 2025) attempts recovery but generates an erroneous image. In contrast, Robust-U1 successfully recovers a clean image that clearly reveals the vehicle’s correct orientation. This case exemplifies that accurate pixel-level recovery is essential for reliable understanding under corruption.

Table 2. Case Study on R-Bench. Red indicate inconsistencies, whereas Green means consistencies with practical scenarios.

Input: Ground Truth (not input):

[Figure 21]

[Figure 22]

Question: Which direction is the vehicle directly in front of us traveling? A. Straight B. Left C. Right (Correct Answer: B. Left)

###### Qwen2.5-VL (Bai et al., 2025):

Let’s analyze the image step by step: 1. The vehicle is positioned in the center of the image, facing the camera. 2. The vehicle appears to be on a straight road with no visible curves or turns. 3. There are no other vehicles directly in front of the vehicle, so we can assume it is the only vehicle in that lane. Thus, the answer is: A. Straight.

Robust-R1 (Tang et al., 2026a): <TYPE> jpeg compression(0.44), dark illumination(0.6) <TYPE_END> <INFLUENCE>The jpeg compression and sharpness change can affect the clarity of the image, making it difficult to discern fine details such as the vehicle’s direction. Dark illumination can obscure the scene, making it challenging to identify objects and their positions accurately.<INFLUENCE_END> <REASONING>The vehicle directly in front of us is positioned on the road, and its direction is not immediately clear due to the image degradation. However, the vehicle appears to be moving straight ahead, as there are no visible signs of turning or changing lanes. <REASONING_END> <CONCLUSION>The vehicle is traveling straight, as indicated by its position on the road and the absence of any visible turning or lane change.<CONCLUSION_END> <ANSWER>A<ANSWER_END>.

BAGEL (Deng et al., 2025): Recovering... Here is the restored image: <IMAGE> (WRONG IN RECOVERY)

<THINKING>For the original image, the vehicle is positioned in the middle of the road, and its orientation suggests it is moving straight ahead. The road markings and the alignment of the vehicle with the lanes support this conclusion. The question asks for the direction, and the options provided are Straight, Left, or Right. Based on the image, the vehicle is traveling straight. For the restored image, I would ensure it is a clean and clear version of the original image,

[Figure 23]

showing no vehicle in the middle of the road. <THINKING_END> The result is A. Ours: (Robust-U1) Recovering... Here is the restored image: <IMAGE>

<THINKING>The image shows a vehicle in motion on a street, with its front facing left, indicating a leftward direction. Despite some image degradation, the vehicle’s orientation is clear, allowing for accurate determination of its travel direction. The vehicle is traveling in the left direction, as indicated by its orientation in the image.<THINKING_END>

[Figure 24]

<ANSWER>B<ANSWER_END>

- Table 3. Quantitative evaluation for anti-degradation on three visual understanding benchmarks (MMMB (Sun et al., 2025), MMStar (Chen et al., 2024a), and RealWorldQA (xAI, 2024)) with three levels of degradation (from 25% to 100%). The best/second best results are shown in Red/Blue, respectively.

MMMB (Sun et al., 2025) MMStar (Chen et al., 2024a) RealWorldQA (xAI, 2024) clean

Category Method

Intensity

Intensity

Intensity

clean

clean

25% 50% 100% 25% 50% 100% 25% 50% 100%

Qwen2.5-VL-3B (Bai et al., 2025) 80.60 79.19 78.68 74.50 54.73 52.90 51.86 48.66 65.22 64.96 63.39 60.65 Gemma3-4B (Team et al., 2025) 71.01 70.30 70.20 69.14 43.93 43.20 42.60 41.33 55.42 54.77 53.72 52.81 InternVL-4B (Chen et al., 2024b) 77.97 77.47 76.66 74.59 51.53 50.26 49.60 46.93 57.38 58.16 57.64 54.90 BAGEL (Deng et al., 2025) 81.92 81.16 80.56 78.48 66.13 64.67 61.33 59.60 68.76 65.75 67.84 63.14

General MLLM

TeCoA (Mao et al., 2023) 57.17 65.71 56.11 51.76 30.46 30.60 30.73 28.06 40.00 39.73 39.47 38.69 Robust CLIP (Schlarmann et al., 2024) 58.83 58.28 57.97 53.33 33.00 32.26 31.80 29.46 43.26 42.48 42.61 41.43 Robust-R1 (Tang et al., 2026a) 81.41 79.49 79.04 75.35 56.86 54.40 53.60 49.53 67.71 66.40 67.05 63.26

Robust MLLM

Ours Robust-U1 84.75 84.14 83.54 83.18 67.20 65.80 64.87 63.87 72.81 72.81 71.50 67.46

Input BAGEL + SFT + RL w. Rpix + RL w. Rsem Ours Ground Truth

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

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

Figure 4. Visual comparison of recovered images across different training stages.

#### 4.2. Performance on Adversarial Corruptions

We further evaluate Robust-U1 under synthetically applied, multi-level adversarial corruptions on three standard VQA benchmarks. Results in Table 3 show that Robust-U1 consistently achieves state-of-the-art performance across all corruption intensities (25%, 50%, 100%).

Main Result. Robust-U1 significantly outperforms both general and robust MLLM baselines. For instance, on MMMB with 100% corruption, it scores 83.18, surpassing the strongest general baseline (BAGEL (Deng et al.,

- 2025), 78.48) and the prior robust SOTA (Robust-R1 (Tang et al., 2026a), 75.35). The advantage is consistent across the more challenging MMStar and RealWorldQA benchmarks. Another key strength of Robust-U1 is its minimal performance drop as corruption severity increases. On MMMB (Sun et al., 2025), accuracy decreases by only 1.57 points from clean to 100% corruption, compared to drops of 3.44 and 6.06 points for BAGEL (Deng et al., 2025) and Robust-R1 (Tang et al., 2026a), respectively. This demonstrates the effectiveness of visual self-recovery in maintaining a reliable understanding under heavy corruption.

#### 4.3. Quality of Visual Recovery

To validate that Robust-U1 can recover high-fidelity visual content, we conduct a comprehensive evaluation using standard image quality metrics: Peak Signal-to-Noise Ratio (PSNR) for pixel-level fidelity, Structural Similarity Index (SSIM) (Hore & Ziou, 2010) for structural preservation, and Learned Perceptual Image Patch Similarity (LPIPS) (Zhang et al., 2018) for perceptual quality. We compare our full model against progressive training stages: the base BAGEL (Deng et al., 2025) model, the model after supervised fine-tuning (SFT), and models refined with individual reinforcement learning rewards.

Quantitative Analysis As shown in Table 5, each stage of our pipeline contributes to improved recovery. SFT establishes a foundational recovery capability, yielding a substantial gain over the base model across all metrics. Subsequent reinforcement learning provides further refinement: optimizing with the pixel-level SSIM reward (Rpix) primarily enhances structural metrics (PSNR, SSIM), while the semantic CLIP reward (Rsem) achieves the best perceptual score (LPIPS), albeit with a minor trade-off in pixel-level accuracy. Critically, our full model, trained with both rewards, achieves the best overall balance.

- Table 4. Ablation study on R-Bench (Li et al., 2024) for MCQ, VQA, and CAP tasks with three degradation levels (from low to high). The best/second best results are shown in Red/Blue, respectively.

Method

MCQ VQA CAP

Overall low mid high low mid high low mid high

Baseline (BAGEL (Deng et al., 2025)) 0.7176 0.6584 0.5793 0.6497 0.6127 0.6150 0.4685 0.4633 0.4288 0.5770 Ours (Robust-U1) 0.7353 0.7329 0.6768 0.7067 0.7164 0.6934 0.8272 0.8059 0.7640 0.7398

w/o Multimodal Reasoning 0.7294 0.6957 0.6524 0.6883 0.6709 0.6281 0.6105 0.6475 0.6378 0.6623 w/o Rpix 0.7059 0.7081 0.6707 0.7025 0.6764 0.6689 0.8340 0.7985 0.7661 0.7257 w/o Rsem 0.7412 0.7205 0.6220 0.6810 0.6873 0.6509 0.8179 0.8278 0.7640 0.7236

- Table 5. Quantitative evaluation of visual recovery quality on Robust-R1 (Tang et al., 2026a) (validation set). The best/second best results are shown in Red/Blue respectively.

Table 6. Qualitative evaluation on multimodal reasoning vs. textbased reasoning. Red indicate inconsistencies, whereas Green means consistencies with practical scenarios.

Method PSNR ↑ SSIM ↑ LPIPS ↓

BAGEL (Deng et al., 2025) 14.37 0.4722 0.5092 + SFT on ImageNet-C (Xie et al., 2020) 20.88 0.6135 0.3444 + RL w. Rpix 21.45 0.6311 0.3299 + RL w. Rsem 21.33 0.6285 0.3233 Ours (Robust-U1) 21.49 0.6314 0.3223

Qualitative Analysis Fig. 4 visually compares the recovery outputs across different stages for three representative corrupted samples. The base BAGEL model yields noisy and distorted outputs. The SFT model removes major artifacts but often lacks fine detail. The outputs from models trained with individual rewards reveal their distinct focuses: Rpix sharpens edges and text, while Rsem better preserves natural textures and color fidelity. Our final Robust-U1 model successfully integrates these complementary advantages, producing clean, sharp, and semantically faithful reconstructions that closely match the ground truth.

#### 4.4. Ablation Study

We conduct comprehensive ablation studies on two key aspects: (1) the effectiveness of multimodal reasoning versus text-based reasoning, and (2) the contributions of the dual rewards in the reinforcement learning stage.

Effectiveness of Multimodal Reasoning Quantitative results in Table 4-(w/o Multimodal Reasoning) confirm the critical role of our proposed multimodal reasoning mechanism. Reasoning without access to the restored image leads to a significant decline in overall performance. The case study in Table 6 illustrates this advantage. When counting objects in a corrupted scene, text-based reasoning yields an incorrect answer, whereas our approach, by recovering the visual content and reasoning over both images, achieves an accurate count, highlighting the superior reliability of direct visual recovery coupled with joint reasoning.

Input: Ground Truth (not input):

[Figure 46]

[Figure 47]

Question: How many buses are there? A. 1 B. 2 C. 3 (Correct Answer: B. 2)

###### Text-based Reasoning:

<THINKING>The buses’ outlines and color patterns remain visible, allowing for the counting of three buses on the left side of the street. The image degradation impacts the clarity, but the buses’ distinct shapes and colors are still discernible, allowing for the correct count of three buses. <THINKING_END> Thus, the answer is: C.

Multimodal Reasoning: (Robust-U1) Recovering... Here is the restored image: <IMAGE>

<THINKING>The image shows a street with two buses visible, despite some noise and compression artifacts. The image’s quality allows for accurate counting of the buses. The image’s quality is compromised by noise and compression, but the buses are still identifiable.<THINKING_END>

[Figure 48]

<ANSWER>B<ANSWER_END>

Effectiveness of Rpix The pixel-level structural reward Rpix is vital for maintaining high visual fidelity in the recovered images, as visually confirmed in Fig. 5. Its absence results in a noticeable drop in performance on tasks that demand precise visual understanding, as shown in Table 4-(w/o Rpix). A different observation is that for the CAP, removing Rpix occasionally leads to marginally better scores, hinting that an over-emphasis on pixel-level perfection might sometimes constrain semantic richness. Nonetheless, the consistent overall performance drop confirms that preserving structural details remains a fundamental requirement for robust visual understanding.

Effectiveness of Rsem The semantic consistency reward Rsem is essential for ensuring the correctness of the recovered image’s content. Ablating this component causes the most severe performance degradation under high levels of

| |
|---|

| |
|---|

|[Figure 49]|
|---|

[Figure 50]

[Figure 51]

|[Figure 52]|
|---|

|w/o 𝐑𝐩𝐢𝐱|
|---|

Ours

- Figure 5. Visual validation of Rpix. Compared with ours (Green), w/o Rpix may produce more artifacts in pixel level (Red)

### References

Bai, S., Chen, K., Liu, X., Wang, J., Ge, W., Song, S., Dang, K., Wang, P., Wang, S., Tang, J., Zhong, H., Zhu, Y., Yang, M., Li, Z., Wan, J., Wang, P., Ding, W., Fu, Z., Xu, Y., Ye, J., Zhang, X., Xie, T., Cheng, Z., Zhang, H., Yang, Z., Xu, H., and Lin, J. Qwen2.5-vl technical report. arXiv, 2025.

Chen, L., Li, J., Dong, X., Zhang, P., Zang, Y., Chen, Z., Duan, H., Wang, J., Qiao, Y., Lin, D., et al. Are we on the right way for evaluating large vision-language models? In NeurIPS, 2024a.

corruption, as shown in Table 4-(w/o Rsem). This pattern indicates that maintaining semantic accuracy becomes increasingly critical as visual degradations worsen. Without Rsem, the model is prone to generating recoveries that are visually coherent but semantically erroneous, which directly misleads the subsequent reasoning process.

### 5. Conclusion

We propose Robust-U1, a novel framework that pioneers a visual self-recovery paradigm for robust multimodal understanding. This approach equips Multimodal Large Language Models with the ability to actively reconstruct clean visual content from corrupted inputs. This explicit reconstruction marks a pivotal advance beyond prior works that rely on implicit feature alignment or textual reasoning alone, establishing a more intrinsic and generalizable form of resilience. By closing the loop from perception to restoration and reasoning, we hope this work opens a path toward building more reliable and robust multimodal systems for future safety-critical applications.

### Acknowledgements

The work was supported by the Research Grants Council of HKSAR under grant number AoE/E-601/24-N. Besides, this work was supported in part by the National Natural Science Foundation of China under Grant 62472359.

### Impact Statement

Our proposed framework, which enables models to selfrecover corrupted visual content, contributes to building more reliable and interpretable vision-language systems. This improvement is particularly relevant for safety-critical applications, such as autonomous navigation and medical image analysis, where performance degradation under realworld noise can have significant consequences. We acknowledge that the capability to reconstruct images could, in principle, be misapplied; we encourage the community to develop guidelines for the ethical use of such technologies.

Chen, Z., Wu, J., Wang, W., Su, W., Chen, G., Xing, S., Zhong, M., Zhang, Q., Zhu, X., Lu, L., et al. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. In CVPR, 2024b.

Chern, E., Hu, Z., Chern, S., Kou, S., Su, J., Ma, Y., Deng, Z., and Liu, P. Thinking with generated images. arXiv preprint arXiv:2505.22525, 2025.

Deng, C., Zhu, D., Li, K., Gou, C., Li, F., Wang, Z., Zhong, S., Yu, W., Nie, X., Song, Z., Shi, G., and Fan, H. Emerging properties in unified multimodal pretraining. arXiv preprint arXiv:2505.14683, 2025.

Duan, H., Yang, J., Qiao, Y., Fang, X., Chen, L., Liu, Y., Dong, X., Zang, Y., Zhang, P., Wang, J., et al. Vlmevalkit: An open-source toolkit for evaluating large multi-modality models. In ACM MM, 2024.

Guo, H., Guo, Y., Zha, Y., Zhang, Y., Li, W., Dai, T., Xia, S.-T., and Li, Y. Mambairv2: Attentive state space restoration. In CVPR, 2025.

Hore, A. and Ziou, D. Image quality metrics: Psnr vs. ssim. In ICPR, pp. 2366–2369. IEEE, 2010.

Hossain, M. Z. and Imteaj, A. Sim-clip: Unsupervised siamese adversarial fine-tuning for robust and semantically-rich vision-language models. arXiv, 2024.

Hu, C., Chen, X., Jia, Z., Shi, W., Zhang, F., Guo, J., and Wei, Y. A semantic decoupling-based two-stage rainy-day attack for revealing weather robustness deficiencies in vision-language models. arXiv preprint arXiv:2601.13238, 2026.

Hurst, A., Lerer, A., Goucher, A. P., Perelman, A., Ramesh, A., Clark, A., Ostrow, A., Welihinda, A., Hayes, A., Radford, A., et al. Gpt-4o system card. arXiv, 2024.

Kong, L., Dong, J., Tang, J., Yang, M.-H., and Pan, J. Efficient visual state space model for image deblurring. In CVPR, pp. 12710–12719, 2025.

Lai, J., Chen, S., Lin, Y., Ye, T., Liu, Y., Fei, S., Xing, Z., Wu, H., Wang, W., and Zhu, L. Snowmaster: Comprehensive real-world image desnowing via mllm with multi-model feedback optimization. In CVPR, pp. 4302– 4312, 2025.

Li, C., Zhang, J., Zhang, Z., Wu, H., Tian, Y., Sun, W., Lu,

- G., Liu, X., Min, X., Lin, W., and Zhai, G. R-bench: Are your large multimodal model robust to real-world corruptions? IEEE JSTSP, 2024.

Liu, H., Li, C., Li, Y., and Lee, Y. J. Improved baselines with visual instruction tuning. In CVPR, 2024.

Liu, J., Jia, Z., Li, J., Li, B., Jin, X., Zeng, W., and Lu, Y. When mllms meet compression distortion: A coding paradigm tailored to mllms. arXiv preprint arXiv:2509.24258, 2025a.

Liu, J., Liu, G., Liang, J., Li, Y., Liu, J., Wang, X., Wan, P., Zhang, D., and Ouyang, W. Flow-grpo: Training flow matching models via online rl. arXiv preprint arXiv:2505.05470, 2025b.

Liu, R., Liu, Z., Tang, J., Ma, Y., Pi, R., Zhang, J., and Chen, Q. Longvideoagent: Multi-agent reasoning with long videos. In ACL, 2026.

Liu, X., Gong, C., and Liu, Q. Flow straight and fast: Learning to generate and transfer data with rectified flow. In ICLR, 2023.

Long, J., Xu, Z., Jiang, T., Yao, W., Jia, S., Ma, C., and Chen, X. Robust sam: On the adversarial robustness of vision foundation models. In AAAI, 2025.

Malik, H. S., Shamshad, F., Naseer, M., Nandakumar, K., Khan, F., and Khan, S. Robust-llava: On the effectiveness of large-scale robust image encoders for multi-modal large language models. In ICCVW, 2025.

Mao, C., Geng, S., Yang, J., Wang, X., and Vondrick, C. Understanding zero-shot adversarial robustness for largescale models. In ICLR, 2023.

Qiu, X., Kan, M., Zhou, Y., and Shan, S. Benchmarking multimodal large language models against image corruptions. In ICCV, pp. 9014–9023, 2025.

Qwen Team. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.

Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., Krueger, G., and Sutskever, I. Learning transferable visual models from natural language supervision. In ICML, 2021.

Schlarmann, C., Singh, N. D., Croce, F., and Hein, M. Robust clip: Unsupervised adversarial fine-tuning of vision embeddings for robust large vision-language models. In ICML, 2024.

Sun, H.-L., Zhou, D.-W., Li, Y., Lu, S., Yi, C., Chen, Q.-G., Xu, Z., Luo, W., Zhang, K., Zhan, D.-C., and Ye, H.J. Parrot: Multilingual visual instruction tuning. arXiv, 2025.

Tang, J., Lu, H., Wu, R., Xu, X., Ma, K., Fang, C., Guo, B., Lu, J., Chen, Q., and Chen, Y.-C. Hawk: Learning to understand open-world video anomalies. In NeurIPS,

- 2024a.

Tang, J., Lu, H., Xu, X., Wu, R., Hu, S., Zhang, T., Cheng, T. W., Ge, M., Chen, Y.-C., and Tsung, F. An incremental unified framework for small defect inspection. In ECCV,

- 2024b.

Tang, J., Wu, R., Xu, X., Hu, S., and Chen, Y.-C. Learning to remove wrinkled transparent film with polarized prior. In CVPR, 2024c.

Tang, J., Chen, J., Wei, W., Xu, X., Liu, R., Wu, X., Xie, Q.,

- Wu, J., Zhang, L., and Chen, Q. Robust-r1: Degradationaware reasoning for robust visual understanding. In AAAI, 2026a.

Tang, J., Xia, Y., Wu, Y.-F., Hu, Y., Chen, Y., Chen, Q.-G., Xu, X., Wu, X., Lu, H., Ma, Y., Lu, S., and Chen, Q. Lpo: Towards accurate gui agent interaction via location preference optimization. In ACL Findings, 2026b.

Tang, J., Yan, Y., Wang, Q., Xia, Y., Geng, B., Chen, J., Ma, K., Zhai, Y., He, Q., Shao, W., Sun, Y., Dai, J., Chen, C., Xu, X., Yao, K., Zhang, L., Wei, W., Chen, Q., Plaza, A., and Zhang, Y. Intelligent remote sensing agents: A survey. Technical Report, 2026c. URL https://github.com/PolyX-Research/ Awesome-Remote-Sensing-Agents.

Team, G., Kamath, A., Ferret, J., Pathak, S., Vieillard, N., Merhej, R., Perrin, S., Matejovicova, T., Ramé, A., Rivière, M., et al. Gemma 3 technical report. arXiv, 2025.

Tian, X., Liao, X., Liu, X., Li, M., and Ren, C. Degradationaware feature perturbation for all-in-one image restoration. In CVPR, 2025.

Wei, J., Wang, X., Schuurmans, D., Bosma, M., Xia, F., Chi, E., Le, Q. V., Zhou, D., et al. Chain-of-thought prompting elicits reasoning in large language models. NeurIPS, 2022.

- Wu, K., Peng, H., Zhou, Z., Xiao, B., Liu, M., Yuan, L., Xuan, H., Valenzuela, M., Chen, X. S., Wang, X., et al. Tinyclip: Clip distillation via affinity mimicking and weight inheritance. In ICCV, pp. 21970–21980, 2023.

xAI. Grok-1.5 vision preview, 2024. URL https://x. ai/blog/grok-1.5v.

Xie, Q., Luong, M.-T., Hovy, E., and Le, Q. V. Self-training with noisy student improves imagenet classification. In CVPR, pp. 10687–10698, 2020.

Zhai, X., Mustafa, B., Kolesnikov, A., and Beyer, L. Sigmoid loss for language image pre-training. In ICCV, 2023.

Zhang, R., Isola, P., Efros, A. A., Shechtman, E., and Wang, O. The unreasonable effectiveness of deep features as a perceptual metric. In CVPR, 2018.

Zhang, Y., Ma, L., Feng, Y., Huang, Z., Zhou, F., and Su, Z. Bilevel layer-positioning lora for real image dehazing. In CVPR, 2026.

Zheng, Z., Yang, M., Hong, J., Zhao, C., Xu, G., Yang, L., Shen, C., and Yu, X. Deepeyes: Incentivizing" thinking with images" via reinforcement learning. arXiv preprint arXiv:2505.14362, 2025.

### Summary of Appendix

This appendix is organized into eight sections, ordered from implementation details to broader discussion:

- • Appendix A – Implementation Details. Per-stage training cost (GPU type, time, memory, trainable parameters) and evaluation protocols on R-Bench and the three anti-degradation benchmarks (MMMB, MMStar, RealWorldQA).
- • Appendix B – Extended Quantitative Comparisons. Comparisons against external restoration modules (DFPIR (Tian et al., 2025), EVSSM (Kong et al., 2025), MambaIRv2 (Guo et al., 2025), BiLaLoRA (Zhang et al., 2026)) plus a discriminative MLLM, and an inference-time analysis with a detect-then-recover variant.
- • Appendix C – Why Self-Recovery Helps Robust Reasoning. Mechanism analyses: isolating reconstruction vs. CoT supervision, decoupling recovery quality (PSNR) from downstream reasoning (R-Bench), and the effect of always-on recovery on clean inputs.
- • Appendix D – Sensitivity Studies. Three training-side robustness studies: a reference-free semantic reward, the reward scaling factor α, and the choice of frozen semantic encoder.
- • Appendix E – Reliability of Recovery and Evaluation. Output-side robustness: hallucination risk analysis (Type I/II/III plus harmful/beneficial/neutral rates) and sensitivity of the reported scores to the LLM-based evaluator.
- • Appendix F – Qualitative Results. Side-by-side visual comparisons of recovery across baselines, and end-to-end case studies that show Robust-U1’s full reasoning trace from corrupted input to final answer.
- • Appendix G – User Study. A controlled human evaluation on R-Bench measuring perceptual preference between BAGEL and Robust-U1 along Semantic Faithfulness and Overall Visual Quality.
- • Appendix H – Limitations and Future Work. Discussion of recovery-quality bounds, dependency on paired training data, and four future directions (efficient architectures, corruption-specific priors, video, and benchmark development).

Together, these materials offer comprehensive evidence for the effectiveness and robustness of the proposed Robust-U1 framework.

### A. Implementation Details

#### A.1. Training Cost

The computational costs of our framework are evaluated using NVIDIA L20 (48GB) GPUs. As detailed in Table 7, the training process is divided into three sequential stages, each with distinct computational requirements, data sources, and trainable parameters. Stage I (reconstruction SFT) dominates the total cost (1920 GPU hours) due to large-scale image– image pair training on ImageNet-C (Xie et al., 2020), while Stages II (RL) and III (joint reasoning) are substantially lighter (160 and 64 GPU hours, respectively). Notably, only Stage III updates both the understanding and generation modules; Stages I–II only optimize the generation module.

- Table 7. Training cost breakdown for the three-stage Robust-U1 framework. All stages are trained on NVIDIA L20 (48GB) GPUs. Stage #GPUs Time GPU Hours Peak Mem. Dataset Trainable Modules

- Stage I (SFT) 64 ∼30 h 1920 42 GB ImageNet-C (Xie et al., 2020) (750k pairs) Generation only

- Stage II (RL) 8 ∼20 h 160 41 GB Robust-R1 (Tang et al., 2026a) training split Generation only

- Stage III (Reasoning) 8 ∼8 h 64 43 GB Robust-R1 (Tang et al., 2026a) reasoning data Understanding + Generation

#### A.2. Evaluation Protocol on R-Bench

To rigorously assess the performance of our model on R-Bench (Li et al., 2024), we implement diverse evaluation protocols tailored to the specific nature of each task:

Multiple-Choice Question (MCQ) For MCQ tasks, we utilize Accuracy as the primary performance indicator to measure the model’s capability in identifying the correct option. The metric is formally defined as:

1 N

Accuracy =

N

I(yi = yˆi), (11)

i=1

where N denotes the total number of test samples, yi represents the ground-truth answer, yˆi is the predicted answer, and I(·) denotes the indicator function.

Visual Question Answering (VQA) and Image Captioning (CAP) Given the open-ended nature of VQA and CAP tasks, we leverage GPT-3.5-turbo (Hurst et al., 2024) as a proxy evaluator to quantify the semantic alignment between model-generated responses and reference answers. The aggregate performance is represented by the mean score:

1 N

Score =

N

si, (12)

i=1

where si denotes the scoring result assigned by GPT-3.5-turbo for the i-th sample. The evaluation framework focuses on three critical dimensions:

- • Completeness: Assessing whether the response encapsulates all essential semantic elements and key points present in the correct answer.
- • Accuracy: Evaluating the factual and logical consistency of the response relative to the ground-truth.
- • Relevance: Measuring the alignment of the response’s content with the intent, context, and core topic of the provided reference.

#### A.3. Evaluation Protocol on Anti-Degradation Benchmarks

In our evaluation across three Anti-Degradation Benchmarks (MMMB (Sun et al., 2025), MMStar (Chen et al., 2024a), and RealWorldQA (xAI, 2024)), we adopt a uniform Multiple-Choice Question (MCQ) format.

Model performance is quantified using the standardized accuracy metric defined in Eq. 11. To ensure robust and accurate extraction of answers from the model’s responses, we employ GPT-3.5-turbo (Hurst et al., 2024) as an automated evaluator through the VLMEvalKit (Duan et al., 2024) framework. Specifically, GPT-3.5-turbo (Hurst et al., 2024) is utilized to parse the model’s output and identify the intended choice label (e.g., A, B, C, D), which is then compared against the ground-truth label to determine the final accuracy.

### B. Extended Quantitative Comparisons

This section reports two extended comparisons that situate Robust-U1 against alternative pipelines: (i) using state-of-theart external restoration models as a preprocessor before a strong discriminative MLLM, and (ii) deploying recovery only when needed via a detect-then-recover variant.

#### B.1. Comparison with External Restoration Modules

To isolate the benefit of internal self-recovery from the more conventional “restoration → understanding” pipeline, we compare Robust-U1 against a strong discriminative MLLM (Qwen2.5-VL-7B (Bai et al., 2025)) preceded by state-ofthe-art external restoration modules. We consider four representative restoration baselines: an all-in-one restoration model DFPIR (Tian et al., 2025), a deblurring model EVSSM (Kong et al., 2025), a denoising model MambaIRv2 (Guo et al., 2025), and a dehazing model BiLaLoRA (Zhang et al., 2026). All baselines are evaluated on R-Bench under the same protocol as Table 1.

As reported in Table 8, all external-restoration variants underperform Robust-U1 by a large margin in overall score, with the best baseline (all-in-one (Tian et al., 2025)) reaching only 0.5511 vs. 0.7398 for Robust-U1. Two factors explain this gap. First, specialized modules (deblurring, denoising, dehazing) require knowing the degradation type and tend to fail under unknown or compound corruptions. Second, even the all-in-one restoration model is optimized for perceptual quality rather than downstream understanding, so the restored images are not necessarily aligned with what the MLLM needs for reasoning. Robust-U1 instead jointly models restoration and understanding via the dual reward and the multimodal reasoning stage, so recovery is directly shaped by what helps the downstream task. This indicates that the gain of Robust-U1 comes from task-aligned, internal recovery, not merely from cleaner images.

- Table 8. Comparison with external restoration modules on R-Bench (Li et al., 2024). Restoration baselines are applied as a preprocessing step before Qwen2.5-VL-7B (Bai et al., 2025). The best results are shown in Red.

Method

MCQ VQA CAP

Overall low mid high low mid high low mid high

all-in-one (Tian et al., 2025) 0.6529 0.6382 0.6280 0.4190 0.3485 0.3503 0.7093 0.6351 0.6336 0.5511 deblurring (Kong et al., 2025) 0.6294 0.5776 0.5671 0.4037 0.3261 0.3240 0.4963 0.6443 0.3559 0.4581 denoising (Guo et al., 2025) 0.6412 0.6787 0.5919 0.4339 0.3630 0.3431 0.7062 0.6303 0.5191 0.5459 dehazing (Zhang et al., 2026) 0.6765 0.6584 0.5671 0.4466 0.3782 0.3371 0.6938 0.5914 0.4845 0.5371

Robust-U1 (Ours) 0.7353 0.7329 0.6768 0.7067 0.7164 0.6934 0.8272 0.8059 0.7640 0.7398

- B.2. Inference Cost and the Detect-then-Recover Variant

We extend the computation-cost analysis in Section 4.3 to inference. We compare three deployment modes on R-Bench using Qwen2.5-VL-7B-class hardware: (i) a standard MLLM that directly answers the question without recovery, (ii) a detect-then-recover pipeline that triggers recovery only when corruption is detected, and (iii) the full Robust-U1 pipeline that always performs recovery and joint reasoning.

- Table 9. Inference-time comparison on R-Bench (Li et al., 2024). “Rec. Mem.” / “Und. Mem.” denote peak GPU memory for the recovery and understanding stages, respectively.

###### Method Rec. Mem. Und. Mem. Rec. Steps Throughput Latency R-Bench

Standard MLLM 0 GB 18 GB 0 – 1.8 s 0.6204 Detect-then-Recover 28 GB 20 GB 50 1.21 step/s 24.6 s 0.7082 Robust-U1 (Ours) 33 GB 23 GB 50 1.21 step/s 55.0 s 0.7398

- Table 9 highlights a clear robustness–cost trade-off. The standard MLLM is fastest but the most fragile under corruption. Detect-then-recover already recovers most of the robustness gap at less than half the latency of the full pipeline, which is a practical option when latency budget is tight. The full Robust-U1 pipeline incurs the highest latency, but yields the strongest robustness. We note that all costs are dominated by the rectified-flow denoising loop (50 steps); reducing the step count or distilling the recovery branch is a promising direction for production deployment.

### C. Why Self-Recovery Helps Robust Reasoning

The previous section quantifies how much Robust-U1 improves over alternative pipelines. This section instead asks why: we (i) isolate the contribution of self-recovery from the additional CoT supervision, (ii) show that downstream reasoning is decoupled from raw recovery quality, and (iii) study the effect of always-on recovery on clean inputs.

#### C.1. Isolating Reconstruction vs. CoT Supervision

A natural concern raised by reviewers is whether the gains of Robust-U1 come specifically from self-recovery or simply from the additional supervision introduced by reconstruction data and chain-of-thought (CoT) reasoning data. To disentangle these two factors, we ablate the two components independently on R-Bench.

Two observations from Table 10 support our central claim. First, adding only CoT supervision improves the BAGEL baseline modestly (0.5770 → 0.6199, +0.0429). Second, adding only task-aligned self-recovery supervision (SFT+RL) yields a noticeably larger improvement (0.5770 → 0.6623, +0.0853), indicating that recovery is the dominant source of gain rather than additional textual supervision. Combining both further pushes the overall score to 0.7398, reflecting the closed-loop synergy between recovery and reasoning. We therefore conclude that the gains of Robust-U1 are not merely a by-product of extra supervision: they are primarily driven by the explicit visual self-recovery mechanism.

- Table 10. Component isolation on R-Bench. “Recon.” indicates whether reconstruction-based recovery supervision (SFT only or SFT+RL) is used, and “CoT” indicates whether reasoning-chain supervision is used.

Variant Recon. CoT R-Bench Overall

BAGEL (Deng et al., 2025) N N 0.5770 Ours (SFT only) SFT N 0.5974 Ours (CoT only) N Y 0.6199 Ours (SFT+RL only) SFT+RL N 0.6623 Robust-U1 (full) SFT+RL Y 0.7398

- C.2. Recovery Quality vs. Downstream Reasoning

A second related question is whether better-looking recovery automatically translates into better downstream reasoning. We answer this by jointly tracking PSNR (a standard reconstruction metric) and R-Bench Overall (the downstream reasoning metric) across progressive training stages.

Table 11. Recovery quality (PSNR) vs. downstream reasoning performance (R-Bench Overall). Method PSNR ↑ R-Bench ↑

BAGEL (Deng et al., 2025) 14.37 0.5770 + SFT 20.88 0.5974 + SFT + RL (only Rpix) 21.45 0.7236 + SFT + RL (only Rsem) 21.33 0.7257 Robust-U1 (full, Rpix + Rsem) 21.49 0.7398

Table 11 reveals a non-trivial decoupling between perceptual fidelity and downstream usefulness. SFT alone already lifts PSNR by +6.51 dB (14.37 → 20.88), but the corresponding R-Bench gain is small (+0.0204), showing that “visually cleaner” recovery is not sufficient. In contrast, adding RL with either reward only marginally improves PSNR (by ≤ 0.6 dB) but produces a large R-Bench jump (∼ +0.13). This indicates that recovery supports reasoning only when it is task-aligned, i.e., when the restoration objective explicitly preserves the semantic cues that downstream understanding depends on, which is exactly what our dual-reward RL is designed to do.

- C.3. Effect of Always-On Recovery on Clean Inputs

In our default deployment, the recovery branch is always executed regardless of the input quality. We quantify the effect of this design on clean inputs to verify that always-on recovery does not harm the model in the absence of corruption.

Table 12. Effect of always-on recovery on clean and corrupted inputs (R-Bench Overall). Input Setting Recovery Enabled? R-Bench Overall

Clean input No 0.7821 Clean input Yes 0.7865 Corrupted input No 0.5605 Corrupted input Yes 0.7398

Two observations from Table 12 are worth highlighting. First, on corrupted inputs, recovery brings a large gain (+0.1793), confirming the central claim of this work. Second, on clean inputs, always-on recovery still provides a small but consistent improvement (+0.0044), suggesting that the recovery module sometimes attenuates residual mild artifacts and produces semantically slightly cleaner visual representations even on nominally clean images. This indicates that always-on recovery is a safe default at deployment time, although a learned gate that conditionally skips recovery on confident clean inputs would further reduce inference cost.

### D. Sensitivity Studies

This section studies how robust Robust-U1 is to three training-time design choices: (i) replacing the paired semantic reward with a reference-free alternative, (ii) varying the reward scaling factor α, and (iii) swapping the frozen semantic encoder.

#### D.1. Reference-Free Semantic Reward

The full Robust-U1 requires paired clean targets to compute both the structural reward Rpix and the semantic reward Rsem. To examine to what extent paired data is necessary, we additionally study a reference-free variant in which Rsem is replaced with an image–text consistency reward: given the original (corrupted) image’s caption T, we measure the cosine similarity between the CLIP text embedding of T and the CLIP image embedding of the recovered image Ir, eliminating the need for a clean reference image.

Table 13. Reference-free semantic reward on R-Bench. “Target GT” indicates whether a paired clean image is required during RL.

Method Target GT Reward R-Bench Overall

No RL (SFT only) – – 0.5770 Reference-free semantic RL No Caption-based CLIP image–text consistency reward 0.6233 Robust-U1 (Ours, full) Yes Structural + semantic reward 0.7398

Table 13 shows that the reference-free variant already improves the SFT-only baseline by +0.0463, indicating that the self-recovery idea is not strictly tied to fully paired training. Nevertheless, our full method with paired structural and semantic supervision still outperforms it by a large margin (+0.1165 overall). This suggests a clear path for extending Robust-U1 to settings where paired clean images are scarce, while also confirming that paired supervision remains the most effective recipe when it is available.

#### D.2. Sensitivity to the Reward Scaling Factor α

The semantic reward Rsem(Ir,Io) = exp −α · (1 − Sim) uses a scaling factor α that controls how aggressively the reward penalizes semantic deviation. We perform a grid search over α ∈ {1,2,5,8,10,15} on the validation set.

Table 14. Sensitivity of Robust-U1 to the semantic reward scaling factor α on R-Bench.

α R-Bench Overall Relative Gap to Best

- 1 0.7352 −0.62%

- 2 0.7371 −0.37%

5 (default) 0.7398 0%

8 0.7383 −0.20% 10 0.7324 −1.00% 15 0.7251 −2.00%

As reported in Table 14, performance varies by less than 0.6% across α ∈ [2,8], indicating that Robust-U1 is robust to the exact choice of α. Smaller values (α ≤ 1) make the reward overly smooth and provide insufficient semantic constraint, while larger values (α ≥ 10) produce a very sharp exponential reward (e.g., α = 15 shrinks the reward from 1 to ≈ 0.22 as Sim drops from 1.0 to 0.9), causing the policy to over-emphasize semantic matching at the expense of pixel-level details (PSNR drops from 21.49 at α = 5 to 20.87 at α = 15). We therefore use α = 5 in all main experiments.

#### D.3. Sensitivity to the Choice of Semantic Encoder

Our semantic reward uses TinyCLIP (Wu et al., 2023) as the frozen encoder. To verify that our conclusions are not tied to this specific encoder, we replace TinyCLIP with three alternatives of different scales and architectures: CLIP-B/16 (Radford et al., 2021), SigLIP-B/16 (Zhai et al., 2023), and a heavily distilled, weaker CLIP (Radford et al., 2021) variant.

Table 15 shows that mainstream CLIP-family encoders give very similar overall scores (within 0.3% of each other), and even a substantially smaller, distilled encoder retains 99.3% of the default performance. The relative ordering across our

Table 15. Sensitivity of Robust-U1 to the choice of frozen semantic encoder used in Rsem. Encoder Parameters R-Bench Overall

TinyCLIP (Wu et al., 2023) (default) 39M 0.7398 CLIP-B/16 (Radford et al., 2021) 149M 0.7376 (−0.30%) SigLIP-B/16 (Zhai et al., 2023) 150M 0.7382 (−0.22%) Distilled weak encoder 9M 0.7345 (−0.72%)

ablations (with vs. without Rsem) is preserved across all encoders. We conclude that the semantic reward is consistently helpful and that Robust-U1 does not depend critically on a specific frozen vision-language encoder.

### E. Reliability of Recovery and Evaluation

The previous sections focus on performance; this section examines two reliability questions: (i) whether the recovery branch introduces hallucinations that mislead downstream reasoning, and (ii) whether the reported scores depend on the specific LLM-based evaluator we use.

#### E.1. Hallucination Risk Analysis

A natural concern is whether the recovery branch occasionally hallucinates content that is not present in the original scene, and whether such hallucinations mislead downstream reasoning. We categorize possible hallucinations into three types in

- Table 16. Table 16. Taxonomy of hallucinations that the recovery branch may introduce.

Type Description Example

- Type I: Structural Objects or shapes not present in the original Blurry circle → human face

- Type II: Semantic Attributes inconsistent with reality Red car restored as blue

- Type III: Over-sharpened Overconfident details from ambiguous input Unreadable license plate → sharp but fabricated

number

To quantify the downstream impact of these hallucinations, we compare the model’s answers when conditioned only on the recovered image Ir against its answers when conditioned on the original clean image Io on R-Bench. We classify each pair into four categories: consistent (same answer), harmful (the recovery flips a correct answer to wrong), beneficial (recovery flips a wrong answer to correct), and neutral (both answers are wrong but different).

- Table 17. Hallucination risk of the recovery branch on R-Bench. Consistency, harmful, beneficial, and neutral rates are computed by comparing answers using the recovered image Ir against answers using the clean image Io.

Setting (Ir vs. Io) Consistency ↑ Harmful ↓ Beneficial ↑ Neutral

Robust-U1 (Ours) 92.3% 4.1% 0.8% 2.8% SFT-only 86.7% 7.2% 1.3% 4.8% BAGEL (Deng et al., 2025) 74.2% 15.6% 2.1% 8.1%

Table 17 shows that the recovery module of Robust-U1 rarely introduces misleading hallucinations: only 4.1% of decisions are flipped from correct to wrong, far below SFT-only (7.2%) and the base BAGEL model (15.6%). The high answer consistency (92.3%) further indicates that, in the vast majority of cases, the recovered image preserves the semantically critical content needed by downstream reasoning. Together with the Rsem ablation in Table 4, these results confirm that semantic supervision is the key mechanism that suppresses hallucinations in the recovery branch.

#### E.2. Sensitivity to the Evaluator Choice

For VQA and CAP tasks on R-Bench, the main paper follows Robust-R1 (Tang et al., 2026a) and uses GPT-3.5-turbo (Hurst et al., 2024) as a proxy evaluator. To verify that our conclusions are not an artifact of this specific evaluator, we re-

score Robust-U1’s outputs on R-Bench with two additional, stronger judges: Qwen3-Max (Qwen Team, 2025) and GPT-4o (Hurst et al., 2024).

Table 18. Sensitivity of Robust-U1’s scores on R-Bench to the choice of LLM-based evaluator.

MCQ VQA CAP

Evaluator

Overall low mid high low mid high low mid high

GPT-3.5-turbo (default) 0.7353 0.7329 0.6768 0.7067 0.7164 0.6934 0.8272 0.8069 0.7640 0.7398 Qwen3-Max (Qwen Team, 2025) 0.7412 0.7205 0.6463 0.6991 0.7118 0.6892 0.8519 0.7957 0.6895 0.7238 GPT-4o (Hurst et al., 2024) 0.7353 0.7143 0.6524 0.7043 0.7050 0.6665 0.7895 0.7970 0.6442 0.7121

As shown in Table 18, all three evaluators agree on the overall trend. The overall score of Robust-U1 varies within a moderate range of 0.7121–0.7398 across evaluators, and remains substantially above all baselines reported in Table 1 regardless of the judge. Qwen3-Max and GPT-4o are slightly stricter than GPT-3.5-turbo, particularly on the harder captioning split, but the relative ordering between methods is preserved. We therefore conclude that the main conclusions of this paper are not sensitive to the specific evaluator.

### F. Qualitative Results

This section provides two complementary kinds of qualitative evidence: side-by-side visual comparisons of recovered images across baselines (Section F.1), and full end-to-end case studies that illustrate Robust-U1’s complete reasoning trace on severely corrupted inputs (Section F.2).

#### F.1. Visual Comparisons of Recovered Images

To further validate the visual self-recovery capability of Robust-U1, we present additional qualitative comparisons in Figure 6. The figure showcases eight representative examples across diverse scenes and corruption types, illustrating the progressive improvement through our training pipeline.

For each example, we show the corrupted input image, followed by the recovered outputs from three models: (1) the base BAGEL (Deng et al., 2025) model, (2) the model after supervised fine-tuning (SFT) only, and (3) our full Robust-U1 model. The ground truth clean image is provided as reference.

Visual inspection reveals consistent patterns: the base BAGEL model often produces blurry reconstructions with residual artifacts, while the SFT model shows clear improvement in removing major corruption effects but may lack fine details. Our full Robust-U1 model generates the most faithful reconstructions, effectively removing compression artifacts, recovering sharp edges, restoring natural textures, and preserving semantic content. These qualitative observations align with the quantitative improvements reported in Section 4.3 and further substantiate that high-quality visual self-recovery serves as a reliable foundation for robust multimodal reasoning.

#### F.2. End-to-End Reasoning Case Studies

To further demonstrate the practical effectiveness and robustness of Robust-U1, we provide end-to-end case studies in Tables 20, 21, 22, and 23. Each case presents a severely corrupted input image alongside the corresponding ground truth (for reference) and a question that requires detailed visual understanding. The examples span multiple scenarios, including traffic signal recognition, object function identification, object counting, and scene understanding.

In every instance, Robust-U1 first explicitly recovers the corrupted image, generating a restored version that clarifies semantic content. The model then utilizes this restored visual information to reason accurately and answer the question. For example, in Table 20, despite severe motion blur, the model correctly identifies that the left-turn arrow is not green. In Table 22, it successfully counts the number of green lights under low-illumination and compression artifacts.

These examples collectively highlight how the explicit visual self-recovery mechanism enables Robust-U1 to reconstruct critical details, such as shapes, colors, and object identities, from heavily degraded inputs. This reconstruction directly supports more reliable and accurate reasoning, confirming that our unified approach effectively mitigates the negative impact of image corruption across diverse real-world tasks and corruption types.

### G. User Study

To complement automated metrics and evaluate the perceptual quality of recovered images, we conducted a controlled user study comparing the outputs of Robust-U1 and the baseline BAGEL model. Automated metrics such as PSNR and SSIM provide objective measures of reconstruction fidelity, but may not fully capture aspects critical for downstream visual understanding, including semantic correctness and perceptual naturalness.

Study Design We recruited 25 participants with backgrounds in computer vision or related fields. Each participant evaluated 15 randomly selected corrupted samples from the R-Bench validation set, covering all three corruption intensity levels and diverse degradation types. For each sample, participants were presented with the corrupted input image alongside the two restored versions (from BAGEL and Robust-U1) in randomized order. Participants were asked to select the preferred restoration based on two independent criteria:

- 1. Semantic Faithfulness: Which recovered image more accurately reconstructs key objects, their attributes, and the overall scene meaning?
- 2. Overall Visual Quality: Which recovered image appears sharper, contains fewer artifacts, and looks more natural?

A "No Preference" option was available for both criteria when participants found the two restorations indistinguishable in quality.

Results and Analysis Aggregated preference percentages across all participants and samples are presented in Table 19. The results demonstrate a strong and consistent preference for images restored by Robust-U1. For Semantic Faithfulness, Robust-U1 was preferred in 92.3% of comparisons, versus 5.6% for BAGEL. For Overall Visual Quality, Robust-U1 was preferred in 85.7% of comparisons, versus 10.1% for BAGEL. The low "No Preference" rates (2.1% and 4.2% respectively) indicate that participants could reliably distinguish between the two restoration methods.

These findings confirm that the improvements measured by automated metrics (Table 5) translate into perceptually significant gains. More importantly, the high preference for Robust-U1 in terms of Semantic Faithfulness aligns with our core hypothesis: high-fidelity visual recovery directly supports more accurate and robust visual understanding by preserving the semantic content essential for reasoning.

Table 19. User study preference results (percentage of comparisons).

Semantic Faithfulness Overall Visual Quality Preferred No Preference Preferred No Preference BAGEL (Deng et al., 2025) 5.6%

Method

10.1%

4.2% Robust-U1 (Ours) 92.3% 85.7%

2.1%

### H. Limitations and Future Work

#### H.1. Limitations

While Robust-U1 demonstrates promising results in enhancing the robustness of Multimodal Large Language Models through visual self-recovery, our work has several limitations that warrant discussion and motivate the future directions discussed in Section H.2.

Recovery Quality. The quality of the recovered images is inherently bounded by the generative capability of the underlying unified MLLM. While our dual-reward RL optimization improves reconstruction fidelity, the model may still struggle with highly complex or severe corruptions where essential visual information is extensively lost. Additionally, the current approach focuses on common real-world corruptions; its performance on rare or adversarial-specific distortions remains to be fully explored.

Dependency on Paired Training Data. Our method requires paired data of corrupted and clean images for the supervised fine-tuning and reinforcement learning stages. While such data can be synthetically generated (as in ImageNet-C (Xie et al.,

2020)), the domain gap between synthetic and real-world corruptions may limit generalization. Moreover, for specialized domains (e.g., industrial defect inspection (Tang et al., 2024b), remote sensing (Tang et al., 2026c), medical imaging, and satellite imagery), obtaining large-scale paired datasets with realistic corruptions is challenging. Section D.1 provides an initial reference-free remedy that partially relaxes this requirement.

#### H.2. Future Work

Building upon the limitations above and the insights from this work, we identify four promising directions for future research:

Efficient Self-Recovery Architectures. Developing more efficient architectures for visual self-recovery is crucial. Future work could explore lightweight recovery modules, knowledge distillation techniques to transfer recovery capabilities to smaller models, or conditional generation mechanisms that require fewer denoising steps—directly addressing the inference-cost trade-off quantified in Section B.2.

Integration with Corruption-Specific Priors. Incorporating explicit models of corruption processes could enhance recovery precision. Future work could develop hybrid approaches that combine data-driven recovery with physics-based or statistical priors of specific corruption types (e.g., deblurring with estimated blur kernels, denoising with noise models). This could be particularly valuable for specialized applications like medical imaging or remote sensing.

Extension to Video and Temporal Domains. Our current work focuses on single-image recovery and reasoning. Extending the self-recovery paradigm to video sequences presents an exciting challenge, where temporal consistency and motion dynamics must be considered. This could enable robust video understanding in adverse conditions like rain, fog, or low-light scenarios.

Benchmark Development for Comprehensive Evaluation. Creating more comprehensive benchmarks covering diverse corruption types, severity levels, and multimodal tasks would facilitate more rigorous evaluation of robust MLLMs. Special emphasis should be placed on real-world, naturally occurring corruptions rather than solely synthetic ones.

By addressing these limitations and pursuing these directions, we believe the paradigm of visual self-recovery can evolve into a fundamental capability for building robust, reliable, and trustworthy multimodal AI systems that operate effectively in the imperfect conditions of the real world.

Input BAGEL + SFT Ours Ground Truth

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

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

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

Figure 6. More visual comparison of recovered images across different baselines.

Table 20. More examples of qualitative evaluation (I).

Input: Ground Truth (not input):

[Figure 98]

[Figure 99]

Question: Is the left turning arrow green?

###### A. Yes B. No (Correct Answer: B. No)

Ours: (Robust-U1) Recovering... Here is the restored image: <IMAGE>

<THINKING>The image’s severe motion blur and compression artifacts make it difficult to discern details like the left-turn arrow. Despite this, the arrow’s position and shape suggest it is not illuminated, indicating it is not green. The image’s degradation makes it challenging to see clear details, but the arrow’s shape and position indicate it is not green. Therefore, the answer is

[Figure 100]

###### B. <THINKING_END>

- <ANSWER>B<ANSWER_END>

Table 21. More examples of qualitative evaluation (II).

Input: Ground Truth (not input):

[Figure 101]

[Figure 102]

Question: What’s the function of the demonstrated object?

- A. Providing entertainment such as movies and music
- B. Offering a variety of food
- C. Transportation of people and cargo.
- D. Offering a variety of drink (Correct Answer: C)

Ours: (Robust-U1) Recovering... Here is the restored image: <IMAGE>

[Figure 103]

<THINKING>The image depicts a train station with visible train tracks and a platform, indicating a transportation function. Despite the blurring, the structure and layout suggest a railway setting, typically linked to people and cargo transport. The blurred image still retains enough context to identify the setting as a train station, which is primarily associated with transportation. Therefore, the function of the demonstrated object is transportation of people and cargo.<THINKING_END>

- <ANSWER>C<ANSWER_END>

Table 22. More examples of qualitative evaluation (III).

Input: Ground Truth (not input):

[Figure 104]

[Figure 105]

Question: How many green lights are in the image? A. 2 B. 3 C. 4 (Correct Answer: B. 3)

Ours: (Robust-U1) Recovering... Here is the restored image: <IMAGE>

<THINKING>The image shows a school bus and a vehicle on a road with traffic lights. Despite some degradation, the green lights are visible. The lights are positioned at the top of the image, and their distinct shape and color make them recognizable even with reduced sharpness and illumination. The scene is dimly lit, but the green lights stand out against the darker background. The image degradation affects the sharpness and illumination, but the green lights are still discernible. By focusing on the visible features and considering the typical placement of traffic lights, it is concluded that there are three green lights in the image.<THINKING_END> <ANSWER>B<ANSWER_END>

[Figure 106]

Table 23. More examples of qualitative evaluation (IV).

Input: Ground Truth (not input):

[Figure 107]

[Figure 108]

Question: Is there a red truck on the left side of the road parked? A. Yes B. No (Correct Answer: B. No)

Ours: (Robust-U1) Recovering... Here is the restored image: <IMAGE>

<THINKING>The question concerns the presence of a red truck on the left side of the road. Despite the image’s degradation, the outline and color of the vehicles on the left are discernible. The vehicles appear to be cars, not trucks, and there is no red truck visible. The image degradation affects the clarity and color recognition, but the vehicles on the left side of the road are identified as cars, not a red truck. Therefore, the answer is No.<THINKING_END>

[Figure 109]

<ANSWER>B<ANSWER_END>

