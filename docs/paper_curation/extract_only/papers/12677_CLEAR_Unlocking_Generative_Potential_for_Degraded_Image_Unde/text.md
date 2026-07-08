# arXiv:2604.04780v1[cs.CV]6Apr2026

## CLEAR: Unlocking Generative Potential for Degraded Image Understanding in Unified Multimodal Models

Xiangzhao Hao1* Zefeng Zhang2* Zhenyu Zhang2† Linhao Yu2 Yao Chen2 Yiqian Zhang2 Haiyun Guo1‡ Shuohuan Wang2 Yu Sun2 1Institute of Automation, Chinese Academy of Sciences 2Baidu Inc.

haoxiangzhao2023@ia.ac.cn

### Abstract

construct MMD-Bench, covering three degradation severity levels across six standard multimodal benchmarks. Experiments show that CLEAR substantially improves robustness on degraded inputs while preserving strong clean-image performance, confirming that the generative and understanding capabilities within unified models can be effectively connected for robust visual understanding. Our code and data are publicly available at https://github. com/haoxiangzhao12138/CLEAR.

Image degradation from blur, noise, compression, and poor illumination severely undermines multimodal understanding in real-world settings. Unified multimodal models that combine understanding and generation within a single architecture are a natural fit for this challenge, as they understand clean images well and their generative pathway can model the fine-grained visual structure that degradation destroys. Yet when directly answering questions about degraded images, these models fail to leverage their own generative capacity. Generation and understanding coexist but remain functionally disconnected. We trace this disconnect to the fact that existing training regimes never ask the model to invoke generation during reasoning, and the standard decode-reencode pathway between the two capabilities does not support effective joint optimization. Together, these prevent answer-level feedback from shaping how the model generates. We present CLEAR, a framework that connects the two capabilities through three progressive steps: (1) supervised fine-tuning on a degradationaware dataset to establish the generate-then-answer reasoning pattern; (2) a Latent Representation Bridge that replaces the decode-reencode detour with a direct, optimizable connection between generation and reasoning; (3) Interleaved GRPO, a reinforcement learning method that leverages this connection to jointly optimize text reasoning and visual generation under answer-correctness rewards. Freed from pixel-level regression targets, the model learns to generate intermediate visual states that not only serve downstream reasoning but also exhibit higher perceptual quality than those produced under explicit reconstruction supervision, revealing that task-driven optimization and visual quality are naturally aligned rather than in conflict. We

[Figure 1]

Multimodal Large models

###### CLEAR

[Figure 2]

[Figure 3]

[Figure 4]

Clean Image

What is he holding in his hand?

What is he holding

[Figure 5]

[Figure 6]

[Figure 7]

in his hand?

[Figure 8]

[Figure 9]

[Figure 10]

He is holding a basketball in his hand.

[Figure 11]

[Figure 12]

This image is a bit blurry, let me restore this image.

[Figure 13]

[Figure 14]

Degrade Image

What is he holding in his hand?

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

He is holding a basketball in his hand.

He is holding an orange.

Figure 1. Top: average scores of commercial and open-source multimodal models on clean versus degraded inputs from MMDBench across six benchmarks. All models show substantial performance drops under degradation. Bottom: comparison between existing multimodal models and CLEAR on a degraded image.

*Equal contribution. †Project lead. ‡Corresponding author.

### 1. Introduction

Image degradation is a routine part of real-world visual data, not an edge case. Images from autonomous driving, surveillance, mobile photography, and video conferencing are frequently corrupted by motion blur, sensor noise, poor illumination, and aggressive compression. These degradations damage the low-level visual cues that multimodal models depend on for recognition, grounding, and reasoning [13, 19]. As Figure 1 illustrates, multimodal models can correctly identify an object in a clean image yet misrecognize it entirely when the same image is degraded. This is not an isolated failure. Across commercial systems such as GPT-4o [29] and open-source architectures of varying scales [8, 23, 38, 48], we observe substantial accuracy losses on degraded versions of six standard benchmarks, indicating that sensitivity to image degradation is a pervasive vulnerability across the current multimodal landscape. Robustness to such degradations is a core requirement for deploying multimodal systems in practice.

Among existing architectures, unified multimodal models stand out for their ability to handle both visual understanding and image generation within a single model. Rather than relying on separate specialist modules, these models share a common backbone across the two tasks [3, 20, 39, 40, 51], with a vision encoder [31, 46] that maps images into semantic features for understanding and a generative pathway that operates through a VAE [16, 32] or discrete tokenizer [11] to produce images from continuous or quantized latent representations. The understanding pathway excels at high-level semantic reasoning, including object recognition, spatial relationship inference, and visual question answering, when the input image is clean. The generative pathway, by contrast, operates at a fundamentally different level of visual representation, capturing lowlevel structure such as texture, edge detail, color distribution, and spatial layout that high-level semantic features tend to discard [47]. Degraded image understanding aims to enable models to interpret images whose low-level visual cues have been unintentionally corrupted, and to answer questions about the high-level semantic information they contain. In unified models, the understanding and generation pathways naturally correspond to these two types of features: the former primarily captures high-level semantics, while the latter models low-level visual details.

Yet when asked to directly answer questions about degraded images, unified models fail to bring these two capabilities together. As the top panel of Figure 1 shows, Bagel [20], Janus-Pro [40], and Emu3 [39] all suffer substantial performance drops under degradation, with no sign that their generative pathway contributes to robustness. The model does not spontaneously invoke generation to compensate for the visual information that degradation has destroyed. Generation and understanding coexist in the same

architecture but remain functionally disconnected. This motivates the central research question of this work: how can we connect generation with the reasoning process to support understanding on degraded images?

To answer this question, we attribute the disconnect to two compounding factors. (1) Behavioral: Existing unified models are never trained to invoke generation as part of the reasoning process for understanding tasks. Their training treats generation and understanding as separate objectives, so the model has no experience with a reasoning pattern that uses generated visual content to support answer production. (2) Structural: Even if such a pattern were introduced, the standard pathway connecting generation to understanding requires that the generated latent representations be decoded into pixel space and re-encoded through a frozen vision encoder before they can influence reasoning. The frozen decoder and encoder sever the computation graph between the generation and understanding stages, preventing gradients from answer-level supervision from propagating back to the parameters that control what the model generates. Taken together, the two factors reinforce each other. Without the behavioral pattern, the model never attempts to generate for understanding, and the structural bottleneck is never even exposed. Without a differentiable optimization route, introducing the behavioral pattern alone cannot teach the model what to generate, only that it should.

To bridge this disconnect, we propose CLEAR (Comprehension via Latent Enhancement and Adaptive Reasoning), a framework that connects the generative and understanding capabilities of unified models through three progressive steps. (1) Behavioral Initialization. We construct a degradation-aware training dataset where samples with mild or no degradation receive direct-answer supervision and samples with severe degradation require the model to first generate an intermediate visual state before answering. Fine-tuning on this dataset teaches the model the generate-then-answer reasoning pattern and establishes when to invoke generation and how to structure the interleaved trajectory. (2) Latent Representation Bridge. With the behavioral pattern in place, the next bottleneck is the decode-reencode pathway. CLEAR addresses this by injecting generated latent representations directly into the reasoning context, eliminating the pixel-space detour entirely. This allows generated visual information to participate in reasoning alongside the original encoded features. At the same time, it creates a direct, differentiable connection from generation to reasoning that makes the joint training in the next step possible. (3) Interleaved GRPO. With the bridge providing an effective optimization route, we apply Interleaved GRPO, a reinforcement learning method building on GRPO [33] and Flow-GRPO [1] that jointly optimizes text reasoning and visual generation within a shared forward pass. The reward centers on final answer correctness,

so answer-level feedback now flows through the bridge to shape how the model generates. Within this training, the model also learns an adaptive generation strategy that evaluates input quality during reasoning and invokes generation only when degradation is likely to impair understanding, avoiding unnecessary computation on clean inputs.

For evaluation, we construct MMD-Bench by applying 16 real-world corruption types at three severity levels to 6 widely used multimodal benchmarks, and additionally evaluate on R-Bench [19], an existing benchmark for degradedimage understanding. Experiments show that CLEAR substantially improves degraded image understanding while maintaining strong clean-image performance. Our analysis further reveals a finding that may seem counter-intuitive. When pixel-level reconstruction supervision is removed and only answer-correctness rewards remain, the model not only preserves but even improves the perceptual quality of its generated intermediate states. This suggests that visual quality is naturally aligned with task optimization, and that explicit reconstruction supervision is more a constraint than a requirement. Our main contributions are as follows.

- • We identify a functional disconnect in unified multimodal models where generation and understanding coexist but fail to cooperate under degraded inputs. To address this, we construct a degradation-aware training set with difficulty-dependent supervision that teaches unified models to invoke generation as part of the reasoning process.
- • We propose CLEAR, which bridges this disconnect through Behavioral Initialization via supervised finetuning, a Latent Representation Bridge that opens a direct optimization route from generation to reasoning, and Interleaved GRPO that jointly optimizes understanding and generation with answer-correctness rewards.
- • Experiments on MMD-Bench and R-bench confirm that CLEAR achieves substantial robustness gains on degraded inputs without sacrificing clean-image performance. Our analysis further shows that removing pixellevel supervision leads to intermediate visual states with higher perceptual quality, indicating that task-driven optimization can naturally aligns with visual quality.

### 2. Related Work

Robustness under Image Degradation. The vulnerability of visual recognition systems to low-level image degradations has been studied extensively since ImageNet-C [13], which showed that modern classifiers suffer substantial accuracy drops under blur, noise, weather effects, and digital distortions. This line of work has since been extended to the multimodal setting. Several benchmarks have been proposed to evaluate vision-language models under degraded conditions, revealing that even models built on strong visual encoders such as CLIP [31] remain highly sensitive to

degraded inputs in tasks including visual question answering, captioning, and multimodal reasoning [12, 19, 49]. As we demonstrate in Section 4, unified multimodal models such as Bagel [20], Janus-Pro [40], and Emu3 [39] suffer comparable drops, indicating that their generative pathways do not spontaneously contribute to robustness. While existing efforts have explored corruption-aware data augmentation [14, 28] and external restoration pipelines [5, 21, 45] to mitigate degradation effects, unified models already possess a generative pathway that operates on exactly the low-level visual structure that degradation destroys, yet this internal capacity is never activated during understanding tasks. Our work focuses on how to connect this capacity to the understanding process so that the model can compensate for degradation from within.

Unified Vision-Language Models. Recent work has moved toward unifying visual understanding and image generation within a single model architecture. Systems such as Chameleon [3] and Emu3 [39] adopt discrete visual tokenization through vector quantization, representing images and text in a shared token space for autoregressive generation of interleaved multimodal sequences. More recent models including Janus [40], Bagel [20], and Transfusion [51] operate on continuous latent representations through variational autoencoders, which preserve richer low-level visual information compared to discrete tokens. Other representative systems such as VILA-U [41], Show-o [42], and Unified-IO 2 [26] explore different tradeoffs between generation quality and understanding performance. A distinctive property of unified architectures is their ability to interleave text generation with image generation, opening the possibility for richer reasoning trajectories than understanding-only models can support. However, this potential remains largely unexplored for visual understanding under degraded conditions. In most current pipelines, generated visual content must be decoded into pixel space and re-encoded by the vision encoder before it can influence subsequent reasoning steps, a procedure that is both computationally expensive and unfavorable for joint optimization. How to effectively route generative representations into the understanding pipeline so that generation actively supports reasoning is the question our work aims to address.

Reinforcement Learning for Vision-Language Reasoning. Reinforcement learning has emerged as a powerful approach for improving reasoning capabilities beyond what supervised fine-tuning can achieve. Methods such as GRPO [33] and DeepSeek-R1 [9] optimize directly for outcome-level rewards, enabling models to discover effective reasoning strategies for mathematical and logical problems without step-level supervision. Recent efforts have begun extending RL to multimodal reasoning, using rulebased or outcome-level rewards to improve visual question answering and grounding [7, 15, 34]. In parallel, diffusion-

based or flow-based policy optimization methods such as DDPO [2], DiffusionDPO [37], and Flow-GRPO [1] apply RL to visual generation under learned reward signals. Despite this progress, existing methods optimize text and image generation in isolation. Unified multimodal models introduce a fundamentally different setting where text and image outputs form a single interleaved trajectory, and the value of generated visual content should be judged by how much it contributes to the final reasoning outcome rather than by its standalone appearance. This requires coordinated optimization where both modalities share a computation graph under a common end-task reward. Our Interleaved GRPO addresses this gap by jointly optimizing text reasoning and visual generation in a single forward pass, with reward centered on answer correctness.

### 3. Method

#### 3.1. Overview

Base Architecture. CLEAR is built on Bagel-7B [20], a unified vision-language model that supports both understanding and generation within a shared Mixture-ofTransformer backbone. The understanding pathway encodes images through a SigLIP [46] vision encoder, while the generation pathway operates through a VAE [16] that maps between pixel space and a continuous latent space. Both pathways feed into the same language model, where generation and understanding share a common reasoning space.

Training Pipeline. As shown in Figure 2, CLEAR trains the model in two stages. Stage 1 performs supervised finetuning on a degradation aware dataset to teach the model the generate-then-answer reasoning pattern and warm-start the Latent Representation Bridge so that the language model can begin reading information from the generated VAE latent. Stage 2 applies Interleaved GRPO, which leverages the bridge as a differentiable connection to jointly optimize text reasoning and visual generation under answer-correctness rewards, during which the model also acquires an adaptive strategy that decides when generation is needed.

Reasoning Trajectory. Given an input, the model first enters an analysis phase within <think>, where it reasons about the visual content and implicitly assesses whether generation would improve its answer. If the model chooses to generate, it emits the <image restore>, which triggers multi-step denoising to produce an intermediate visual state in VAE latent space. The resulting latent tokens are injected directly into the reasoning context through the Latent Representation Bridge, serving as the visual input for subsequent reasoning. Rather than decoding to pixel space and re-encoding through a vision encoder, the model reasons directly over the generated latent representation, performing what we term latent reasoning. Another analysis phase then

processes these latent tokens alongside the preceding text to produce the final answer within <answer>. When the model judges that the available visual information is sufficient, it skips generation entirely and proceeds directly to the answer, keeping the trajectory compact.

#### 3.2. Behavioral Initialization through SFT

The first step addresses the behavioral gap. Existing unified models have never been trained to invoke generation as part of the reasoning process for understanding tasks. We bridge this gap through supervised fine-tuning on a purpose-built degradation-aware dataset.

Training Data Construction. We sample a subset from the LLaVA-OneVision [18] instruction-tuning dataset. For each sampled image, we apply degradations drawn from a pool of 16 corruption types covering four categories (capture, transmission, environment, and post-processing) at three intensity levels, and then evaluate whether the base Bagel model can correctly answer the associated question on the degraded version. The full list of corruption types and their severity parameters are provided in the supplementary material C. Samples that the model answers correctly are assigned the direct-answer pathway, while samples it fails on are assigned the generate-then-answer pathway. For both types, we use GPT-4.1 [30] to generate structured reasoning traces, with direct-answer traces containing analysis and answer phases and generate-then-answer traces additionally containing the generation trigger and post-generation analysis. All traces are filtered against ground-truth answers to remove incorrect reasoning. The final SFT dataset contains 24k samples, split evenly between the two pathway types. A separate non-overlapping set of 24k samples is reserved for the Interleaved GRPO stage. Since LLaVA-OneVision is the same corpus used to train the base Bagel model, any potential overlap with evaluation benchmarks affects all compared methods equally.

Training Objective. The core objective is next-token prediction over the text tokens in the trajectory (LCE), which teaches the model the interleaved reasoning format and the conditions under which generation should be triggered. Two auxiliary losses support the visual generation side. An MSE loss (LMSE) provides an initial training signal for the denoising process by encouraging the generated VAE latent to approximate the clean image in latent space. A distillation loss (LKL) uses the ViT features of the clean image as the teacher signal to guide the VAE latent representations. Since the language model has been pretrained exclusively with ViT features as visual input, raw VAE latent tokens fall outside the representation distribution it can interpret. The KL loss addresses this by encouraging the VAE latent hidden states to move toward the ViT feature distribution at each transformer layer, with higher layers receiving greater weight. This does not collapse the two representations into

- Stage 1: Supervised Fine-tuning

- Stage 2: Interleave GRPO

[Figure 25]

[Figure 26]

Distill loss

MSE loss

Ce loss

Ce loss

[Figure 27]

###### Unified Multimodal Models

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

What is he holding in his hand?

<think> This image is a bit blurry … </think>

<answer> He is holding a

<think> I can see the

System

image clearly now …

Prompt

<image_restore>

basketball

</think>

Degraded Image

</answer>

Question

Analyze Denoising VAE Cond ViT cond Answer

Analyze

Group Outputs

GRPO Flow-GRPO GRPO

Group Relative

| |Policy Optimization| |
|---|---|---|
| |ℒInterleaved = ℒGRPO + 𝜆ℒFlow−GRPO| |
| |Advantages<br><br>Rewards<br><br>···<br><br>···<br><br>|𝐴1|
|---|
<br><br>|𝑟1|
|---|
<br><br>𝐴𝑖 =<br><br>𝑟𝑖 − mean 𝑟1 𝑟2 ⋯ 𝑟𝐺 st𝑑 𝑟1 𝑟2 ⋯ 𝑟𝐺<br><br>|𝑟2|
|---|
<br><br>|𝑟𝐺|
|---|
<br><br>|𝐴2|
|---|
<br><br>|𝐴𝐺|
|---|
| |

###### …

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

Next Token Prediction

Latent Representation Bridge

Next Token Prediction

Unified Multimodal Models

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

<think> This image is a

<answer> He is

[Figure 55]

<think> I can see the

bit blurry … </think>

holding a

VAE Latent

What is he holding

image clearly now …

<image_restore>

basketball </answer>

in his hand?

</think>

Degraded

Analyze Denoising Analyze

Answer

Image Question

Figure 2. Overview of CLEAR. Stage 1 (top) performs supervised fine-tuning to establish the generate-then-answer reasoning pattern and warm-start the Latent Representation Bridge, with both VAE latent and ViT re-encoded features injected during this stage. Stage 2 (bottom) applies Interleaved GRPO, where text tokens are optimized with GRPO and the denoising step with Flow-GRPO, sharing the same grouprelative advantage from answer-correctness rewards. The ViT path is removed in Stage 2, making the bridge the sole connection between generation and reasoning.

Decode Reencode

Latent Representation Bridge

identical outputs. Rather, it teaches the language model to read useful information from the VAE latent path while the VAE representations retain their characteristic low-level structural content that ViT features lack. The overall objective is

VAE

decoder

|× 𝑁|
|---|

|× N|
|---|

###### UMM UMM

[Figure 56]

LSFT = LCE + λMSE LMSE + λKL LKL. (1)

[Figure 57]

[Figure 58]

VAE encoder

ViT encoder

Denoising

Denoising

During SFT, both the VAE latent and the ViT re-encoded features of the generated image are injected into the reasoning context after denoising. The ViT re-encoded features serve as an auxiliary input that supports the KL distillation loss and provides the model with a familiar representation format during the early stages of bridge training; they are removed in the GRPO stage once the bridge is established (Section 3.4). The SigLIP vision encoder and VAE encoder/decoder remain frozen throughout. Only the language model backbone is updated.

Figure 3. Left: the standard decode-reencode path in existing unified models. The generated VAE latent must be decoded into pixels and re-encoded through the ViT before it can enter the reasoning context. Right: the Latent Representation Bridge in CLEAR. The generated VAE latent is directly concatenated into the reasoning context, eliminating the decode-reencode bottleneck and providing an effective optimization route from answer correctness back to generation.

After SFT, the model has learned when to generate and how to structure the interleaved trajectory, but what it generates remains constrained by the MSE target. While the clean-image latent provides a reasonable initialization for the denoising process, the MSE objective suffers from a well-known regression-to-mean tendency [17, 27] that limits the sharpness and perceptual quality of the generated states. To move beyond this ceiling, the model needs a training signal that connects generation directly to answer correctness, which is what the next two steps provide.

#### 3.3. Latent Representation Bridge

The second step addresses the structural barrier that prevents generation from being jointly optimized with understanding.

As illustrated in Figure 3 (left), existing unified models route generated visual content through a lengthy detour before it can participate in reasoning. The VAE latent produced by the denoising process must first be decoded into pixel space, then re-encoded through the vision encoder,

before the resulting features can enter the language model context. This path adds substantial computational cost and, more importantly, severs the gradient connection between generation and reasoning, because the frozen decoder and encoder sit between the two stages and block backpropagation.

CLEAR replaces this detour with a direct connection, as shown in Figure 3 (right). After denoising, the generated VAE latent tokens are concatenated into the reasoning context alongside the original ViT features and text tokens. This gives the language model two complementary sources of visual evidence for reasoning: high-level semantic information from the ViT features of the degraded input and fine-grained structural detail from the generated VAE latent.

The more critical consequence is for training. Because the generated latent now participates directly in the computation that produces the answer, answer-level supervision can reach the generation process through a differentiable path. This is what makes joint optimization in the next step possible. During SFT, the KL distillation loss has already provided a warm start for this connection so that the language model can begin exploiting information from the VAE latent tokens. In the GRPO stage that follows, the ViT re-encoding route used during SFT is removed, and the bridge becomes the sole connection between generation and reasoning. This ensures that answer-correctness rewards flow entirely through the bridge, freeing the generation process from pixel-level regression targets and allowing it to be shaped by downstream task performance.

#### 3.4. Interleaved GRPO

After SFT, the model can produce generate-then-answer trajectories, and the bridge provides a differentiable path from generation to reasoning. The missing piece is a training signal that connects answer correctness to the generation process, so that the model learns to generate visual states that actually help it answer rather than simply approximate clean images under an MSE objective. Interleaved GRPO fills this role by jointly optimizing text reasoning and visual generation under answer-correctness rewards.

Background. GRPO [33] optimizes a language model by sampling a group of G completions for each input, computing group-relative advantages from their rewards, and updating the policy with a clipped surrogate loss that increases the probability of higher-reward completions. FlowGRPO [1] extends this idea to flow matching models by converting deterministic ODE sampling into an equivalent SDE [22, 24] to introduce the stochasticity that GRPO requires, and deriving per-step transition log-probabilities from the predicted velocity field so that the same clipped surrogate structure can be applied to denoising steps.

Challenge of Joint Optimization. In our setting, each trajectory interleaves text tokens and a multi-step denoising

process within a single autoregressive sequence, and we need to optimize both modalities under a shared reward. Naively combining the two objectives would require maintaining the full computation graph across all N denoising steps for each of the G sampled trajectories, which is prohibitive in GPU memory since each denoising step involves a full forward pass through the model backbone.

Trajectory Sampling and Training. We address this through two design choices that reduce the cost of imageside optimization to a tractable level.

For trajectory sampling, we generate G complete interleaved sequences per input. The text portion of each trajectory is sampled autoregressively as in standard GRPO. For the denoising portion, each trajectory uses SDE-based sampling to generate a single denoising trajectory of N steps, recording the state pair (xt,xt+∆t) at each step without retaining the computation graph. The reward Ri for each trajectory is computed from the final answer, and the grouprelative advantage Aˆi is derived across the G trajectories.

For the training forward pass, we randomly select one denoising step from the N recorded states for each trajectory and inject the corresponding noisy latent xt into the model input at its original position in the sequence. The model then performs a single forward pass over the full interleaved sequence, simultaneously producing text logits at all text positions and the predicted velocity field vθ at the selected denoising position. This reduces the image-side optimization from N forward passes per trajectory to one, making the memory and compute cost comparable to standard text-only GRPO with only one additional token position per sequence.

From the text logits, we compute the standard GRPO loss:

LGRPO = −E min ri,t · Aˆi, clip(ri,t,1−ϵ,1+ϵ) · Aˆi ,

(2)

###### where ri,t = πθ(oi,t|q,oi,<t)/πθ

(oi,t|q,oi,<t) is the pertoken importance ratio. From the predicted velocity field, we compute the transition log-probability under the SDE formulation and obtain the Flow-GRPO loss:

old

LFlow-GRPO = −min rimg · Aˆi, clip(rimg,1−ϵ,1+ϵ) · Aˆi ,

(3)

###### where rimg = exp(log pθ(xt+∆t|xt) − log pθ

(xt+∆t|xt)) is the transition probability ratio at the selected denoising step. The final Interleaved GRPO loss combines both:

old

LInterleaved = LGRPO + λLFlow-GRPO. (4)

Because both losses are derived from the same forward pass and share hidden representations, gradients from the GRPO

Table 1. Main results under Hard degradation. R-Bench-Dis is an existing degraded-image benchmark; the remaining six are from MMDBench. Best in bold, second best underlined. †Closed-source results are included as reference points and are not directly comparable due to differences in model scale and training data.

MMD-Bench (Hard) Method MMBench MM-Vet MMVP CV-Bench MMStar RealWorldQA R-Bench-Dis AVG Closed-source models†

GPT-4o-mini 67.02 50.91 64.00 59.87 45.93 58.95 61.21 58.27 GPT-4.1-mini 76.08 51.88 71.00 74.96 60.73 72.41 72.52 68.51 Gemini-2.5-Flash 79.33 66.55 72.33 76.01 62.00 69.15 72.72 71.16

Open-source unified models

Emu3 53.71 21.51 65.00 58.34 42.06 52.55 55.15 49.76 Janus-Pro 55.57 31.33 52.66 66.75 41.53 43.52 49.09 48.64 Bagel 67.88 45.09 65.66 64.81 55.53 58.43 61.64 60.15

CLEAR variants (Bagel backbone)

Text-only CoT 63.62 48.30 70.33 64.18 56.93 53.98 62.82 60.02 CLEAR-SFT 72.06 47.56 70.33 70.51 57.67 60.13 65.65 63.42 CLEAR-RL 72.52 51.97 71.33 72.25 60.67 61.05 67.07 65.26

loss influence the image generation pathway through the bridge, and gradients from the Flow-GRPO loss influence textual reasoning through the shared attention mechanism. Critically, both objectives use the same advantage Aˆi derived from a single reward, ensuring that text reasoning and visual generation are optimized toward the same goal. By selecting only one denoising step per trajectory, the training forward pass adds minimal memory overhead beyond standard text-only GRPO while still coupling the two modalities within a shared computation graph.

Reward Design. The reward combines three components. The dominant term Racc measures final answer correctness, evaluated by an external language model following the LLM-as-judge paradigm [50] on a continuous scale. Rfmt encourages valid output structure by checking for properly formed analysis and answer blocks. Rdec evaluates the generation decision retrospectively: it assigns higher rewards when the model generated before answering correctly and penalizes cases where the model skipped generation and answered incorrectly; the remaining two cases (generated but answered incorrectly, or skipped generation and answered correctly) receive a neutral reward. This encourages the model to invoke generation when it would help while not penalizing correct decisions to skip. No reward component targets the perceptual quality of the generated visual state. The overall reward is

###### R = wacc Racc + wfmt Rfmt + wdec Rdec. (5)

Adaptive Generation Strategy. The decision reward Rdec, combined with the natural mixture of generate-then-answer and direct-answer trajectories in the sampled completions, gives rise to an input-dependent generation policy. During the analysis phase, the model implicitly evaluates whether generation would improve its answer and decides whether to emit the <image restore> token. This is not a sep-

arate classifier or a manually designed threshold, but a behavior shaped by the reward signal within the Interleaved GRPO framework. As we show in Section 4, the model learns to generate more frequently as degradation severity increases and to largely skip generation on clean inputs, achieving a favorable balance between robustness and efficiency.

### 4. Experiments

#### 4.1. Implementation Details

CLEAR is built on Bagel-7B [20] with a SigLIP [46] vision encoder and a Qwen2-based [43] language model backbone. Only the language model backbone is updated; the SigLIP encoder, VAE encoder, and VAE decoder remain frozen. The SFT dataset contains 24k samples split evenly between direct-answer and generate-then-answer trajectories, constructed from LLaVA-OneVision [18] as described in Section 3.2. We train SFT for 3 epochs with learning rate 2e-5, loss weights λMSE=0.5 and λKL=0.1, and ViT token drop probability 0.4. For Interleaved GRPO, we use a separate 24k-sample set with group size G=4, learning rate 1e-6, ϵ=0.2, image loss weight λ=0.3, and reward weights wacc=0.75, wfmt=0.1, wdec=0.15. Denoising uses 30 steps. All experiments run on 8 NVIDIA A100 80GB GPUs.

We evaluate on MMD-Bench, which applies 16 corruption types at three severity levels (detailed in the supplementary material B) to six benchmarks: MMBench [25], MM-Vet [44], MMVP [36], CV-Bench [35], MMStar [6], and RealWorldQA, plus R-Bench-Dis [19] as an existing degraded-image benchmark.All evaluations are conducted using VLMEvalKit [10].

#### 4.2. Main Results

Table 1 presents the main results under Hard degradation. We highlight three key observations.

##### (1) Degradation vulnerability is universal. All mod-

els suffer substantial accuracy losses under degradation regardless of architecture and scale. Even GPT-4.1-mini and Gemini-2.5-Flash show notable drops compared to their clean-image performance (Figure 1). Among open-source unified models, Emu3, Janus-Pro, and Bagel all degrade significantly, confirming that existing generative pathways do not spontaneously contribute to robustness.

##### (2) Verbal reasoning cannot compensate for visual in-

formation loss. Text-only CoT provides no meaningful advantage over the base model (60.02 vs 60.15), with scattered gains on some benchmarks offset by regressions on others (e.g., MMBench 63.62 vs 67.88), indicating that finegrained visual information destroyed by degradation cannot be recovered through language-level reasoning alone.

##### (3) Connecting generation to reasoning yields sub-

stantial gains. CLEAR-SFT improves the average by 3.27 points over Bagel with consistent gains across all benchmarks. CLEAR-RL pushes this further to 65.26, the best result among all open-source models on all seven evaluation sets. The gain from SFT to RL is most pronounced on MMVet (47.56 → 51.97) and MMStar (57.67 → 60.67), confirming the value of Interleaved GRPO for benchmarks requiring multi-cue reasoning. Overall, CLEAR-RL improves Bagel by 5.11 points (8.5% relative) within the same architecture without additional parameters or external modules.

##### (4) Comparison with external restoration. Although

CLEAR addresses the internal capability disconnect within unified models rather than competing with external restoration pipelines, we provide a reference comparison on RBench [19], an independently constructed degraded-image benchmark. A restoration model [4] followed by Bagel reaches 65.05, improving over the base Bagel (61.64) but still falling behind CLEAR-RL (67.07) by 2.02 points. The restoration model optimizes for pixel-level fidelity without coupling to the downstream reasoning task, whereas CLEAR shapes its generated states end-to-end through answer-correctness rewards, producing intermediate representations that better serve understanding.

#### 4.3. Robustness Analysis

A natural question is whether CLEAR’s gains under degradation simply reflect an overall quality improvement from fine-tuning or a genuine increase in robustness. Table 2 addresses this by comparing the performance drop from clean to hard inputs.

Bagel loses 7.29 points (10.9% relative) under hard degradation. CLEAR-RL reduces this drop to 5.56 points (7.8%), a 24% reduction in the robustness gap. The improvement on clean images reflects the benefit of the struc-

- Table 2. Robustness analysis. Clean and Hard scores are averaged over the six MMD-Bench benchmarks. Drop = Clean − Hard.

Method Clean Hard Drop (↓)

Bagel 66.86 59.57 7.29 CLEAR-SFT 69.34 63.04 6.30 CLEAR-RL 70.27 64.96 5.31

- Table 3. Component ablation averaged over six MMD-Bench benchmarks. “Dec-reenc” replaces the bridge with the standard decode-reencode path during GRPO.

Configuration Clean Hard Bagel (base) 66.86 59.57 + SFT 69.34 63.04 + SFT + Dec-reenc + GRPO 70.14 63.72 + SFT + Bridge (w/o GRPO) 69.51 63.11 + SFT + Bridge + GRPO 70.27 64.96

- Table 4. No-reference perceptual quality and reasoning accuracy of intermediate visual states. BRISQUE and NIQE (lower is better) measure distortion; MUSIQ (higher is better) measures overall quality.

State BRISQUE↓ NIQE↓ MUSIQ↑ Hard AVG↑

SFT state 43.73 5.32 42.63 63.04 RL state 41.53 4.93 45.74 64.96

tured reasoning format shared by all fine-tuned variants, while the narrower degradation gap demonstrates the additional contribution of the generative pathway. CLEAR’s advantage over Bagel also widens as degradation severity increases, from +4.11 on clean to +5.39 on hard (full severitylevel results in the supplement), directly confirming that the generative pathway provides increasing benefit when degradation is most severe.

#### 4.4. Ablation Studies

Table 3 validates the necessity of each progressive step by systematically removing components.

Applying GRPO directly to the base Bagel model without SFT is not feasible, because Bagel has never been trained to produce generate-then-answer trajectories. Without the behavioral pattern established by SFT, the model does not emit the <image restore> token or structure its output in the interleaved format that GRPO requires, leaving the reinforcement learning process without valid trajectories to optimize.

SFT alone yields a 3.47% gain on hard inputs, demonstrating that the generate-then-answer pattern is valuable even without joint optimization. Replacing the bridge with decode-reencode during GRPO limits the gain to 63.72, because the frozen decoder and encoder block answer-level credit from reaching the generation process. The bridge without GRPO performs comparably to SFT alone (63.11

###### Easy Degraded Hard Degraded

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

What's the function of the demonstrated object?

Options:\nA. running\nB. Play football\nC. Play tennis\nD. Play basketball

[Figure 63]

[Figure 64]

<think> Image condition diagnosis: The image is clear, with good lighting and no significant degradation …

[Figure 65]

[Figure 66]

<think> The image is heavily blurred, making it difficult to

<think> The restored image shows a person

on a field with a netted fence … </think>

###### </think>

clearly identify the objects … </think> <image_restore>

- <answer> A. The front of the vehicle is on the left hand side of the photo </answer>

[Figure 67]

[Figure 68]

Which direction is

this vehicle facing?

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

<think> The image is affected by noise, which makes it

difficult to discern fine details, but the shapes and positions of the dogs are still visible.… </think>

- <answer> B. The smaller dog is facing to the left </answer>

<answer> D. Play basketball </answer>

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

Is this a one-way road?

Which way is the smaller

Options:\nA. Yes\nB. No

dog facing?

Please select the correct answer from the options above.

[Figure 81]

[Figure 82]

<think> The restored image is clear and

[Figure 83]

[Figure 84]

<think> The image is heavily blurred, making it difficult

shows a street with cars parked on both sides and a crosswalk … </think>

to see specific details …

<answer> B. No </answer>

###### </think> <image_restore>

- Figure 4. Qualitative examples of CLEAR’s adaptive reasoning. Left: on a mildly noisy image, the model skips generation and answers directly. Right: on a severely blurred image, the model triggers generation to recover obscured details before answering.

MMBench MMVeT MMVP CV-Bench MMStar RealWorldQA

0

10

20

30

40

50

GenerationTriggeringRate(%)

1.2

4.7

8.4

4.8

2.1

10.2

5.5

11.6

16.8

13.2

7.6

18.4

24.6

35.3

46.6

41.6

28.5

41.7

Degradation Level Low

| |
|---|

| |
|---|

Mid

| |
|---|

High Inference Time

0

20

40

60

80

100

InferenceTime(min)

54m

61m

95m

7m 7m

9m

5m 6m

8m

15m

19m

31m

28m

31m

43m

13m

17m

27m

- Figure 5. Generation triggering rate (bars, left axis) and total inference time (line, right axis) across degradation severity levels for each benchmark.

rate rises monotonically from 5.2% at low to 12.2% at mid and 36.4% at high, with MMVP and RealWorldQA reaching the highest rates (46.6% and 41.7%) due to their reliance on fine-grained visual detail. Inference time closely tracks the triggering rate: at low degradation, evaluation time remains near the base model, while under high degradation the additional denoising cost raises time in proportion to the fraction of samples that trigger generation. The overhead is thus determined by the adaptive policy rather than any fixed per-input cost, confirming that CLEAR concentrates computation on inputs where generation yields the largest accuracy benefit.

Intermediate Visual States. A central claim of this work is that pixel-level reconstruction supervision constrains rather than helps the generation process. To test this, we evaluate no-reference perceptual quality metrics on samples that triggered generation under hard degradation (Table 4). During SFT, the MSE loss encourages generated states to approximate the clean image, but its well-known regression-tomean tendency produces perceptually smooth outputs that score poorly on sharpness and texture metrics. After Interleaved GRPO, pixel-level supervision is removed entirely and generation is driven solely by answer-correctness rewards. Despite receiving no perceptual quality signal, RL states consistently outperform SFT states across all three metrics, because the visual properties that help reasoning, sharp edges for reading text, clear textures for identifying objects, well-defined structure for spatial reasoning, are precisely those that no-reference metrics also value. A pure task-driven reward therefore simultaneously improves both reasoning accuracy and perceptual quality, confirming that the two objectives are naturally aligned and that explicit reconstruction supervision acts as a constraint rather than a requirement.

vs 63.04), confirming that its value lies in enabling joint optimization rather than providing a better inference-time representation. The full pipeline achieves the best result on both clean and hard inputs, with each component building on the previous one.

Beyond validating each component, we further examine whether the generated VAE latent provides visual information that the degraded input alone cannot supply. We keep the full generate-then-answer trajectory intact but replace the generated latent at the bridge with the degraded image’s own VAE latent, preserving the same token format, count, and position in the reasoning context. Under this substitution the hard-degradation average drops from 64.96 to 62.06. Since the reasoning structure remains identical and only the latent content differs, this gap confirms that the denoising process recovers visual structure absent from the degraded input and that the model’s subsequent reasoning actively relies on this recovered information.

#### 4.5. Analysis

Adaptive Generation Behavior and Inference Overhead. Figure 5 shows the generation triggering rate and total inference time across degradation levels. The average triggering

Qualitative Examples. Figure 4 illustrates CLEAR’s reasoning in two contrasting scenarios. On a mildly noisy im-

age, the model judges that the available visual information is sufficient, skips generation entirely, and answers directly. On a severely blurred image, the first analysis phase identifies that critical visual details are unreadable, triggers generation, and the post-generation phase extracts recovered information for a correct answer.

### 5. Conclusion

We identified a functional disconnect in unified multimodal models where generation and understanding coexist but remain isolated under degraded inputs, and proposed CLEAR to bridge this gap. Through supervised finetuning that establishes the generate-then-answer reasoning pattern, a Latent Representation Bridge that opens a direct optimization route from generation to reasoning, and Interleaved GRPO that jointly optimizes both capabilities under answer-correctness rewards, CLEAR enables unified models to leverage their own generative capacity for robust visual understanding. Experiments on MMD-Bench show that CLEAR substantially improves degraded-image performance while preserving clean-image accuracy, with the model learning to invoke generation selectively based on input quality. Our analysis further reveals that removing pixel-level reconstruction supervision and relying solely on answer-correctness rewards leads to intermediate visual states with higher perceptual quality, not lower, confirming that task-driven optimization and visual clarity are naturally aligned and that explicit reconstruction targets act as a constraint rather than a requirement.

### References

- [1] Improving generation quality of flow-based multimodal models via grpo. arXiv preprint, 2025. 2, 4, 6, 13
- [2] Kevin Black, Michael Janner, Yilun Du, Ilya Kostrikov, and Sergey Levine. Training diffusion models with reinforcement learning. arXiv preprint arXiv:2305.13301, 2023. 4
- [3] Chameleon Team. Chameleon: Mixed-modal early-fusion foundation models. arXiv preprint arXiv:2405.09818, 2024. 2, 3
- [4] Liangyu Chen, Xiaojie Chu, Xiangyu Zhang, and Jian Sun. Simple baselines for image restoration, 2022. 8
- [5] Liangyu Chen, Xiaojie Chu, Xiangyu Zhang, and Jian Sun. Simple baselines for image restoration. In European Conference on Computer Vision (ECCV), 2022. 3
- [6] Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Jiaqi Wang, Yu Qiao, Dahua Lin, and Feng Zhu. Are we on the right way for evaluating large vision-language models? arXiv preprint arXiv:2403.20330, 2024. 7, 15
- [7] Liang Chen, Qiguang Bai, Kanzhi Xu, Jiahao Li, et al. R1-v: Reinforcing super generalization ability in vision language models with less than $3. arXiv preprint arXiv:2503.01785,

2025. 3

- [8] Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, Bin Li, Ping Luo, Tong Lu, Yu Qiao, and Jifeng Dai. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 24185–24198, 2024. 2
- [9] DeepSeek-AI. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025. 3
- [10] Haodong Duan, Junming Yang, Yuxuan Qiao, Xinyu Fang, Lin Chen, Yuan Liu, Xiaoyi Dong, Yuhang Zang, Pan Zhang, Jiaqi Wang, et al. Vlmevalkit: An open-source toolkit for evaluating large multi-modality models. In Proceedings of the 32nd ACM International Conference on Multimedia, pages 11198–11201, 2024. 7, 15
- [11] Patrick Esser, Robin Rombach, and Bj¨orn Ommer. Taming transformers for high-resolution image synthesis. In CVPR,

2021. 2

- [12] Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin, Mengdan Zhang, Xu Lin, Jinrui Yang, Xiawu Zheng, Ke Li, Xing Sun, Yunsheng Wu, and Rongrong Ji. Mme: A comprehensive evaluation benchmark for multimodal large language models. arXiv preprint arXiv:2306.13394, 2023. 3
- [13] Dan Hendrycks and Thomas Dietterich. Benchmarking neural network robustness to common corruptions and perturbations. Proceedings of the International Conference on Learning Representations, 2019. 2, 3
- [14] Dan Hendrycks, Norman Mu, Ekin D Cubuk, Barret Zoph, Justin Gilmer, and Balaji Lakshminarayanan. Augmix: A simple data processing method to improve robustness and uncertainty. In ICLR, 2020. 3
- [15] Wenyi Huang, Enfang Feng, Yufei Gao, et al. Vision-r1: Incentivizing reasoning capability in multimodal large language models. arXiv preprint arXiv:2503.06749, 2025. 3
- [16] Diederik P Kingma and Max Welling. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114, 2013. 2, 4
- [17] Christian Ledig, Lucas Theis, Ferenc Husz´ar, Jose Caballero, Andrew Cunningham, Alejandro Acosta, Andrew Aitken, Alykhan Tejani, Johannes Totz, Zehan Wang, and Wenzhe Shi. Photo-realistic single image super-resolution using a generative adversarial network. In CVPR, 2017. 5
- [18] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Yanwei Li, Ziwei Liu, and Chunyuan Li. Llava-onevision: Easy visual task transfer. arXiv preprint arXiv:2408.03326, 2024. 4, 7, 15
- [19] Chunyi Li, Jianbo Zhang, Zicheng Zhang, Haoning Wu, Yuan Tian, Wei Sun, Guo Lu, Xiaohong Liu, Xiongkuo Min, Weisi Lin, and Guangtao Zhai. R-bench: Are your large multimodal model robust to real-world corruptions? arXiv preprint arXiv:2410.05474, 2024. 2, 3, 7, 8, 14
- [20] Kunchang Li et al. Emerging properties in unified multimodal pretraining. arXiv preprint arXiv:2505.14683, 2025. 2, 3, 4, 7
- [21] Jingyun Liang, Jiezhang Cao, Guolei Sun, Kai Zhang, Luc Van Gool, and Radu Timofte. Swinir: Image restoration using swin transformer. In ICCVW, 2021. 3

- [22] Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. In ICLR, 2023. 6, 13
- [23] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. In NeurIPS, 2023. 2
- [24] Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. In ICLR, 2023. 6, 13
- [25] Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, Kai Chen, and Dahua Lin. Mmbench: Is your multi-modal model an all-around player? arXiv preprint arXiv:2307.06281, 2023. 7, 14
- [26] Jiasen Lu, Christopher Clark, Sangho Lee, Zichen Zhang, Savya Khosla, Ryan Marten, Derek Hoiem, and Aniruddha Kembhavi. Unified-io 2: Scaling autoregressive multimodal models with vision, language, audio, and action. CVPR,

2024. 3

- [27] Michael Mathieu, Camille Couprie, and Yann LeCun. Deep multi-scale video prediction beyond mean square error. In ICLR, 2016. 5
- [28] Eric Mintun, Alexander Kirillov, and Saining Xie. On interaction between augmentations and corruptions in natural corruption robustness. In NeurIPS, 2021. 3
- [29] OpenAI. Gpt-4o system card. arXiv preprint arXiv:2410.21276, 2024. 2
- [30] OpenAI. Gpt-4.1. https://openai.com/index/ gpt-4-1/, 2025. 4, 15
- [31] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In International Conference on Machine Learning, pages 8748–8763. PMLR,

2021. 2, 3

- [32] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In CVPR, 2022. 2
- [33] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Mingchuan Zhang, Y.K. Li, Y. Wu, and Daya Guo. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024. 2, 3, 6, 13
- [34] Haozhan Shen, Zilun Zhang, Qian Zhao, Ruochen Zhang, et al. Vlm-r1: A stable and generalizable r1-style large vision-language model. arXiv preprint arXiv:2504.07615,

2025. 3

- [35] Shengbang Tong, Ellis Brown, Penghao Wu, Sanghyun Woo, Manoj Middepogu, Sai Charitha Akula, Jihan Yang, Shusheng Yang, Adithya Iyer, Xichen Pan, et al. Cambrian1: A fully open, vision-centric exploration of multimodal llms. arXiv preprint arXiv:2406.16860, 2024. 7, 15
- [36] Shengbang Tong, Zhuang Liu, Yuexiang Zhai, Yi Ma, Yann LeCun, and Saining Xie. Eyes wide shut? exploring the visual shortcomings of multimodal llms. CVPR, 2024. 7, 15
- [37] Bram Wallace, Meiqi Dang, Rafael Rafailov, Linqi Zhou, Aaron Lou, Senthil Purber, Stefano Ermon, Caiming Xiong,

- Shafiq Joty, and Nikhil Naik. Diffusion model alignment using direct preference optimization. 2024. 4
- [38] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Yang Fan, Kai Dang, Mengfei Du, Xuancheng Ren, Rui Men, Dayiheng Liu, Chang Zhou, Jingren Zhou, and Junyang Lin. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024. 2
- [39] Xinlong Wang, Xiaosong Zhang, Zhengxiong Luo, Quan Sun, Yufeng Cui, Jinsheng Wang, Fan Zhang, Yueze Wang, Zhen Li, Qiying Yu, et al. Emu3: Next-token prediction is all you need. arXiv preprint arXiv:2409.18869, 2024. 2, 3
- [40] Chengyue Wu, Xiaokang Chen, Zhiyu Wu, Yiyang Ma, Xingchao Liu, Zizheng Pan, Wen Liu, Zhenda Xie, Xingkai Yu, Chong Ruan, and Ping Luo. Janus: Decoupling visual encoding for unified multimodal understanding and generation. arXiv preprint arXiv:2410.13848, 2024. 2, 3
- [41] Yecheng Wu, Zhuoyang Zhang, Junyu Chen, Haotian Tang, Dacheng Li, Yunhao Fang, Ligeng Zhu, Enze Xie, Hongxu Yin, Li Yi, Song Han, and Yao Lu. Vila-u: a unified foundation model integrating visual understanding and generation. arXiv preprint arXiv:2409.04429, 2024. 3
- [42] Jinheng Xie, Weijia Mao, Zechen Bai, David Junhao Zhang, Weihao Wang, Kevin Qinghong Lin, Yuchao Gu, Zhijie Chen, Zhenheng Yang, and Mike Zheng Shou. Show-o: One single transformer to unify multimodal understanding and generation. arXiv preprint arXiv:2408.12528, 2024. 3
- [43] An Yang, Baosong Yang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Zhou, et al. Qwen2 technical report. arXiv preprint arXiv:2407.10671, 2024. 7
- [44] Weihao Yu, Zhengyuan Yang, Linjie Li, Jianfeng Wang, Kevin Lin, Zicheng Liu, Xinchao Wang, and Lijuan Wang. Mm-vet: Evaluating large multimodal models for integrated capabilities. arXiv preprint arXiv:2308.02490, 2023. 7, 14
- [45] Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang. Restormer: Efficient transformer for high-resolution image restoration. In CVPR, 2022. 3
- [46] Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023. 2, 4, 7
- [47] Junyi Zhang, Charles Herrmann, Junhwa Hur, Luisa Polania Cabrera, Varun Jampani, Deqing Sun, and Ming-Hsuan Yang. A tale of two features: Stable diffusion complements dino for zero-shot semantic correspondence. Advances in Neural Information Processing Systems, 36:45533–45547,

2023. 2

- [48] Zefeng Zhang, Xiangzhao Hao, Hengzhu Tang, Zhenyu Zhang, Jiawei Sheng, Xiaodong Li, Zhenyang Li, Li Gao, Daiting Shi, Dawei Yin, et al. Cooper: A unified model for cooperative perception and reasoning in spatial intelligence. arXiv preprint arXiv:2512.04563, 2025. 2
- [49] Changqian Zhao et al. Evaluating the robustness of multimodal large language models against image corruptions. arXiv preprint, 2024. 3

- [50] Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. Judging llm-as-a-judge with mt-bench and chatbot arena. In NeurIPS, 2024. 7
- [51] Chunting Zhou, Lili Yu, Arun Babu, Kushal Tirumala, Michihiro Yasunaga, Leonid Shamis, Jacob Kahn, Xuezhe Ma, Luke Zettlemoyer, and Omer Levy. Transfusion: Predict the next token and diffuse images with one multi-modal model. arXiv preprint arXiv:2408.11039, 2024. 2, 3

### 6. Appendix

This supplementary material is organized as follows. Appendix A provides detailed derivations of the GRPO and Flow-GRPO objectives that underlie Interleaved GRPO. Appendix B describes the construction of MMD-Bench, including the 16 corruption types and six base benchmarks. Appendix C details the training data construction pipeline and reasoning trace generation process. Appendix D presents the system prompt shared across training and inference. Appendix E reports full severity-level results. Appendix F provides per-corruption analysis. Appendix G analyzes inference latency. Appendix H examines hyperparameter sensitivity. Appendix I gives additional reward design details. Appendix J presents reasoning trace examples and additional qualitative results.

### A. GRPO and Flow-GRPO Details

This section provides the full computation process of GRPO and Flow-GRPO, which are combined into Interleaved GRPO in Section 3.4 of the main text.

#### A.1. GRPO

Group Relative Policy Optimization [33] eliminates the need for a separate value network by estimating advantages from a group of sampled completions. For each input query q, the model samples a group of G completions {o1,o2,...,oG} from the current policy πθ

. Each completion oi is scored by a reward function to obtain Ri.

old

The group-relative advantage for the i-th completion is computed by normalizing rewards within the group:

Ri − mean(R1,R2,...,RG) std(R1,R2,...,RG)

Aˆi =

. (6)

This relative normalization ensures that the advantage reflects how good a completion is compared to its peers from the same input, rather than in absolute terms.

The policy is then updated by maximizing the clipped surrogate objective. For each token t in completion oi, the per-token importance ratio is:

πθ(oi,t | q,oi,<t) πθ

. (7) The GRPO objective is:

ri,t =

(oi,t | q,oi,<t)

old

|oi|

G

1 G

1 |oi|

min ri,t · Aˆi,

JGRPO(θ) =

t=1

i=1

clip(ri,t,1−ϵ,1+ϵ) · Aˆi − β DKL[πθ∥πref],

(8)

where ϵ is the clipping range that prevents excessively large policy updates, and the KL divergence term with coefficient β regularizes the updated policy to stay close to a

reference policy πref, preventing reward hacking. Letting ρi,t = πref(oi,t | q,oi,<t)/πθ(oi,t | q,oi,<t), the KL divergence is estimated per token as:

DKL[πθ∥πref] ≈ ρi,t − log ρi,t − 1. (9)

The key advantage of GRPO over PPO is that no critic network is needed. The group-relative advantage estimation replaces the learned value baseline with a simple statistical normalization over the sampled group, significantly reducing memory consumption and implementation complexity.

#### A.2. Flow-GRPO

Flow-GRPO [1] extends GRPO to flow matching models, which generate images through a learned velocity field that transports samples from noise to data along a continuoustime trajectory.

Flow Matching Background. In rectified flow [22, 24], a velocity field vθ(xt,t) is learned to transport a noise sample x1 ∼ N(0,I) to a data sample x0 along a straight path. The sampling process follows the ODE:

###### dxt = vθ(xt,t)dt, (10)

where t decreases from 1 (pure noise) to 0 (clean image). This deterministic process generates images by discretizing the ODE into N steps.

ODE-to-SDE Conversion. GRPO requires stochastic sampling to generate diverse trajectories for advantage estimation. Since the flow ODE is deterministic, Flow-GRPO converts it into an equivalent SDE that preserves the same marginal distribution pt(x) at all timesteps. Using the Fokker-Planck equation to match marginal densities, the equivalent reverse-time SDE is:

σt2 2 ∇log pt(xt) dt + σt dw, (11)

dxt = vθ(xt,t) +

where σt is a noise schedule that controls the level of stochasticity and dw is a Wiener process. The marginal score ∇log pt(x) is related to the velocity field by:

xt + (1 − t)vθ(xt,t) t

. (12)

∇log pt(x) = −

Substituting this into the SDE and applying EulerMaruyama discretization yields the update rule:

σ2 2 ·

xt + (1 − t)vθ t

∆t

xt+∆t = xt + vθ +

(13)

√

∆t ϵ, ϵ ∼ N(0,I).

+σ

Transition Log-Probability. The SDE update defines a Gaussian transition distribution. Letting sθ = −(xt + (1 − t)vθ)/t denote the score estimate, the predicted mean of the next state is:

2

2 sθ ∆t, (14)

µθ = xt + vθ − σ

and the transition log-probability is:

- 1

- 2σ2∆t ∥xt+∆t − µθ∥2+C, (15)

log pθ(xt+∆t | xt) = −

where C is a constant independent of θ that cancels in the importance ratio.

Policy Update. Analogous to GRPO, Flow-GRPO samples G denoising trajectories for each input, computes the reward for each, and derives the group-relative advantage Aˆi. The importance ratio at a denoising step is:

###### rimg = exp(log pθ(xt+∆t | xt) − log pθ

old

(xt+∆t | xt)),

- (16)

and the Flow-GRPO objective follows the same clipped surrogate structure:

LFlow-GRPO = −min rimg · Aˆi, clip(rimg,1−ϵ,1+ϵ) · Aˆi .

- (17)

- A.3. From Separate to Interleaved

Standard GRPO operates on text-only sequences, while Flow-GRPO operates on image-only denoising trajectories. In our setting, each trajectory contains both text tokens and a denoising process interleaved within a single autoregressive sequence. The challenge is that these two objectives operate on fundamentally different token types (discrete text tokens vs. continuous latent states) yet must share the same reward signal.

Interleaved GRPO addresses this by making three design choices. First, both objectives share the same group-relative advantage Aˆi computed from a single reward that evaluates the final answer, ensuring that text reasoning and visual generation are optimized toward the same goal. Second, only one denoising step per trajectory is selected for optimization during training, reducing the memory cost from N forward passes to one while still coupling the two modalities through shared hidden representations. Third, the Latent Representation Bridge provides the differentiable connection that allows gradients from the text-side GRPO loss to reach the generation parameters and gradients from the Flow-GRPO loss to influence text reasoning. The full formulation is presented in Section 3.4 of the main text.

- B. MMD-Bench Details

- B.1. Motivation and Comparison with R-Bench

R-Bench [19] is the most closely related existing benchmark for evaluating multimodal models under image degradation. While R-Bench provides a valuable testbed, it has several limitations that motivate the construction of MMDBench.

First, R-Bench uses a fixed set of pre-degraded images without providing clean counterparts or systematic severity

control. This makes it difficult to measure the performance drop caused by degradation or to analyze how model robustness changes as severity increases. MMD-Bench addresses this by applying degradations to existing benchmarks whose clean-image performance is already well established, enabling direct computation of the clean-to-degraded gap.

Second, R-Bench does not organize its degradation types into structured categories that reflect real-world degradation sources, making it harder to diagnose which types of degradation a model is most vulnerable to. MMD-Bench groups 16 corruption types into four categories (capture, transmission, environmental, and post-processing), enabling systematic analysis at both the category level and the individual corruption level.

Third, R-Bench evaluates a single combined score without separating the contributions of different visual capabilities. By building on six established benchmarks that each target different aspects of multimodal understanding, MMD-Bench enables fine-grained diagnosis of which capabilities are most affected by degradation.

Table 5 summarizes the key differences.

Table 5. Comparison between R-Bench and MMD-Bench.

Property R-Bench MMD-Bench Clean reference available No Yes Severity levels Single Three (Low/Mid/Hard) Structured categorization No Four categories # base benchmarks 1 6 Capability diagnosis Combined score Per-benchmark

We include R-Bench-Dis as an additional evaluation set in our experiments (Table 1) to demonstrate that CLEAR generalizes to independently constructed degraded-image benchmarks beyond our own MMD-Bench.

#### B.2. Base Benchmarks

MMD-Bench is constructed by applying 16 real-world corruption types at three severity levels (Low, Mid, Hard) to six widely used multimodal benchmarks. The six base benchmarks are selected to collectively cover a broad spectrum of multimodal understanding capabilities, from coarse-grained perception to fine-grained reasoning.

MMBench [25] is a bilingual benchmark containing 2,974 multiple-choice questions that span 20 fine-grained ability dimensions organized into three hierarchical levels covering both perception and reasoning. It employs a CircularEval strategy that rotates the answer option order across multiple passes to reduce position bias, providing more reliable evaluation results than standard single-pass accuracy.

MM-Vet [44] evaluates the integrated capabilities of multimodal models across six core vision-language dimensions: recognition, knowledge, OCR, spatial awareness, language generation, and math. It contains 218 open-ended

questions over 200 images, and uses GPT-4 as an automated judge to score free-form responses, making it particularly suitable for evaluating complex answers that require multiple capabilities simultaneously.

MMVP [36] is designed to probe visual perception failures that stem from CLIP-based vision encoders. It consists of 300 image pairs that appear similar to CLIP but differ in visually obvious ways to humans, paired with straightforward yes/no questions. MMVP is especially relevant to our study because the visual distinctions it tests are precisely the kind of fine-grained cues that degradation tends to destroy.

CV-Bench [35] contains 2,638 manually inspected examples repurposed from classic computer vision benchmarks including ADE20K, COCO, and Omni3D. It assesses multimodal models on traditional vision tasks such as object detection, counting, and depth estimation within a VQA format, focusing on vision-centric spatial understanding that demands accurate low-level perception.

MMStar [6] comprises 1,500 carefully curated samples designed to ensure visual dependency and minimal data leakage. It evaluates six core capabilities across 18 detailed axes, with each sample verified to be unanswerable without the visual input, making it a rigorous test of genuine multimodal reasoning rather than language-only shortcuts.

RealWorldQA consists of 764 images captured from real-world scenarios including driving scenes and everyday environments, each paired with a question about spatial relationships or scene understanding. It tests practical visual comprehension in naturalistic settings where image quality is inherently variable.

Table 6 summarizes the key characteristics of each benchmark.

Table 6. Base benchmarks used in MMD-Bench.

Benchmark # Samples Primary Focus Evaluation MMBench 2,974 Fine-grained multi-ability assessment Accuracy (CircularEval) MM-Vet 218 Integrated VL capability evaluation GPT-4 scoring MMVP 300 CLIP-blind visual perception Accuracy CV-Bench 2,638 Vision-centric spatial understanding Accuracy MMStar 1,500 Vision-indispensable reasoning Accuracy RealWorldQA 764 Real-world spatial comprehension Accuracy

- Table 7 lists all 16 corruption types organized by cat-

egory, and Figure 6 visualizes representative examples at each severity level.

For each corruption type, we define three severity levels that progressively increase the degradation strength. Low degradation introduces mild perturbations that are noticeable but do not severely affect content understanding. Mid degradation produces clearly visible artifacts that begin to impair fine-grained recognition. Hard degradation substantially obscures visual details, making many low-level cues unrecoverable without additional information. The severity parameters for each corruption type are calibrated empirically to ensure that these qualitative descriptions hold con-

Table 7. The 16 corruption types in MMD-Bench, organized into four categories that reflect distinct real-world degradation sources.

Category Corruption Types Real-world Source Capture lens blur, lens flare, motion blur, Camera hardware and

dirty lens, hsv saturation shooting conditions Transmission jpeg compression, block exchange, Lossy compression and

mean shift, scan lines bandwidth limitations Environmental dark illumination, atmospheric turbulence, Adverse lighting and

gaussian noise, color diffusion atmospheric conditions Post-processing sharpness change, graffiti, Downstream editing and watermark damage overlay artifacts

sistently across different image contents.

For each benchmark, all test images are corrupted with all 16 types at all three severity levels, yielding 48 degraded variants per image. Evaluation follows the original benchmark protocols, with the only modification being the replacement of clean images with their degraded counterparts. All evaluations are conducted using VLMEvalKit [10] to ensure reproducibility. The reported score at each severity level is the average across all 16 corruption types, providing a comprehensive measure of robustness rather than sensitivity to any single degradation.

### C. Training Data Construction

This section provides additional details on the construction of the degradation-aware SFT dataset and the GRPO training set described in Section 3.2.

#### C.1. Data Collection Pipeline

We sample 48k image-question pairs from the LLaVAOneVision [18] instruction-tuning dataset, selecting samples that cover diverse visual domains including natural scenes, documents, charts, and everyday objects. For each sampled image, we randomly select one of the 16 corruption types and one of the three severity levels to generate a degraded version. We then query the base Bagel model with the degraded image and the associated question to determine whether the model can answer correctly. Samples that the model answers correctly are assigned to the directanswer pathway, while samples it fails on are assigned to the generate-then-answer pathway. We balance the two pathways to a 1:1 ratio by subsampling the larger group. The final 48k samples are split into two non-overlapping sets of 24k each, one for SFT and one for Interleaved GRPO.

#### C.2. Reasoning Trace Generation

For both pathway types, we use GPT-4.1 [30] to generate structured reasoning traces. The generation prompt provides GPT-4.1 with the clean image, the degraded image, the question, and the ground-truth answer, and instructs it to produce a trace conforming to one of two patterns depending on the assigned pathway. Figure 7 shows the prompt

[Figure 85]

- Figure 6. Visualization of all 16 corruption types at three severity levels. Each row shows one corruption type applied to the same source image at Low (left), Mid (center), and Hard (right) severity.

template for the generate-then-answer pathway.

For the direct-answer pathway, Steps B and C are omitted. The prompt instructs GPT-4.1 to diagnose the image condition, determine that the visual information is sufficient despite mild degradation, and proceed directly to reasoning

and answering.

All generated traces are filtered against ground-truth answers. Traces whose final answers do not match the ground truth are discarded and regenerated up to three times before the sample is dropped entirely.

|Prompt Template for Generate-then-Answer Trace<br><br>You are an advanced AI training data generator. You will be given a degraded image, its clean version, a question, and the ground-truth answer. Your task is to synthesize a high-quality reasoning trace that follows the structure below.<br><br>Step A (Diagnosis): Act as if seeing only the corrupted image. Describe the visual defects you observe. Hypothesize the degradation type. State that the quality is too poor to answer confidently and decide to invoke the restoration tool. Do NOT reveal or guess the answer at this stage.<br>Step B (Tool Trigger): Output <image restore> on its own line.<br><br>Step C (Post-restoration Analysis): Act as if you have received the restored image. Confirm that the previously observed artifacts are resolved. Locate the visual details that are now visible and relevant to answering the question. Connect these details to form a conclusion.<br>Step D (Answer): Provide the final answer concisely, matching the ground-truth answer.<br>|
|---|

- Figure 7. Prompt template used to generate reasoning traces for the generate-then-answer pathway via GPT-4.1.

C.3. Dataset Statistics

Table 8 summarizes the key statistics of the final SFT dataset.

Table 8. SFT dataset statistics.

Property Value

Total samples 24,886 Direct-answer samples 12,267 Generate-then-answer samples 12,619 Average trace length (direct) 606 Average trace length (generate) 1080 corruption types used 16 severity levels 3 Source dataset LLaVA-OneVision GRPO set (separate, non-overlapping) 24,480

The degradation distribution in the SFT dataset is approximately uniform across the 16 corruption types and three severity levels, with minor imbalances arising from the pathway assignment process since harder corruptions are more likely to cause model failures and thus be assigned to the generate-then-answer pathway.

D. System Prompt

- Figure 8 shows the system prompt used throughout training (both SFT and Interleaved GRPO) and inference. The prompt defines two reasoning scenarios: Scenario 1 for

the generate-then-answer pathway when degradation obscures critical details, and Scenario 2 for direct answering when visual information is sufficient despite degradation. It specifies the output structure with <think>, <answer>, and <image restore> tags, and requires the model to perform explicit image quality analysis before deciding whether to invoke generation.

### E. Full Severity-Level Results

Table 9 reports the complete results of CLEAR-RL across all severity levels on each of the six MMD-Bench benchmarks.

Performance degrades gracefully from Clean to Hard, with the average dropping from 70.27 to 64.97 (a 5.30-point or 7.5% relative decline). The drop from Clean to Low is modest (0.88 points), indicating that CLEAR-RL handles mild degradation with minimal accuracy loss. The steepest decline occurs between Mid and Hard (3.40 points), where severe corruptions begin to obscure critical visual details beyond what the generative pathway can fully recover. Across benchmarks, MM-Vet shows the largest absolute drop from Clean to Hard (9.22 points), consistent with its reliance on integrated multi-cue reasoning where multiple visual details must be simultaneously recovered. RealWorldQA is notably stable across severity levels (61.43 to 61.05), likely because its spatial reasoning questions depend more on scene layout than on fine texture details.

### F. Per-Corruption Analysis

To understand how CLEAR-RL performs across different degradation sources, we report accuracy for each of the 16 corruption types under Hard degradation, grouped by their four categories (Table 7). Table 10 compares Bagel with CLEAR-RL, averaged over the six MMD-Bench benchmarks.

CLEAR-RL improves over Bagel consistently across all 16 corruption types. At the category level, capture degradations benefit the most (+5.98), as blur and flare destroy fine spatial structure that the generative pathway is particularly well-suited to recover. Environmental degradations show the second largest gain (+5.75), where noise and poor illumination uniformly obscure texture and color details across the image. Transmission degradations gain +5.28, with compression artifacts partially recoverable through the learned denoising trajectory. Post-processing degradations benefit the least (+4.19), likely because corruptions such as graffiti and watermark overlay foreign content that is structurally different from natural image degradation, making them harder to address through the same generative process.

At the individual corruption level, motion blur (+7.17) and gaussian noise (+6.41) show the largest improvements. Both corruptions uniformly degrade spatial structure across

|System Prompt<br><br>You are a specialized multimodal assistant. Your purpose is to solve visual question answering tasks by thinking step-by-step and utilizing an image restoration tool when necessary.<br><br>Skills. You can trigger image restoration by generating the following special token sequence: <image restore>. This tool performs enhancement operations (e.g., deblurring, denoising) on the input image to reveal details that are currently obscured.<br><br>Instruction.<br><br>(1) Reasoning (<think>): In each turn, you must start with a <think> tag. Inside, conduct a step-by-step reasoning process. Analyze image quality by identifying degradations (blur, noise, low resolution, etc.). Assess sufficiency by determining if the current image quality allows you to answer the question confidently.<br>(2) Tool Usage: If the degradation prevents you from seeing critical details required for the answer, you MUST trigger the restoration tool by outputting <image restore>. If the answer is visible despite the degradation, do NOT use the tool.<br><br>(3) Answering (<answer>): After reasoning (and potential restoration), provide your final response in the <answer> tag. The answer should be natural, concise, and direct.<br>(4) Format: Keep your output compact. Avoid unnecessary newlines between tags.<br><br><br>Scenario 1 (restoration needed): <think> The image is heavily blurred, making the text unreadable. I need to restore it to extract the information. </think> <image restore> <think> The restored image is clear. The text says “EXIT”. </think> <answer> The text on the sign is “EXIT”. </answer><br><br>Scenario 2 (direct answer): <think> Although there is some noise, the red car is clearly visible in the foreground. </think> <answer> The car is red. </answer><br>|
|---|

Figure 8. System prompt used during SFT, Interleaved GRPO, and inference. The same prompt is shared across all stages without modification.

- Table 9. CLEAR-RL results across degradation severity levels on each MMD-Bench benchmark. AVG is computed over the six benchmarks.

Level MMBench MM-Vet MMVP CV-Bench MMStar RealWorldQA AVG Clean 80.03 61.19 77.00 76.14 65.86 61.43 70.27 Low 79.41 57.93 75.33 75.86 64.40 63.39 69.39 Mid 78.48 55.27 75.00 75.17 64.60 61.69 68.37 Hard 72.52 51.97 71.33 72.25 60.67 61.05 64.97

the entire image, creating exactly the type of low-level information loss that the generative pathway is designed to recover. lens blur (+6.66) and dark illumination (+6.31) follow closely, as these similarly destroy fine detail in a spatially uniform manner. In contrast, graffiti (+3.58) shows the smallest gain. Unlike natural degradations that reduce image quality uniformly, graffiti overlays spatially localized foreign content onto the image, and the denoising process must distinguish between original content and overlaid artifacts, a fundamentally harder task than recovering information that has been blurred or noised. color diffusion (+4.51) and sharpness change (+4.77) also show moderate gains, as these corruptions alter global image properties in ways that partially preserve the structural cues the understanding pathway can still exploit, reducing the marginal benefit of generation.

This fine-grained analysis confirms that the generate-

then-answer strategy is broadly effective rather than specialized to any particular degradation type, while also revealing that the generative pathway is most beneficial when degradation uniformly destroys spatial structure and least beneficial when corruptions introduce foreign visual content.

### G. Inference Latency

The main text (Figure 5) shows that inference time closely tracks the generation triggering rate. Table 11 reports the evaluation time of CLEAR-RL on each benchmark across degradation levels, measured on a single NVIDIA A100 80GB GPU.

Inference time increases monotonically with degradation severity across all benchmarks, driven by the higher generation triggering rate. Under Low degradation, the average triggering rate is only 5.2% and total evaluation time across the six MMD-Bench benchmarks is 2 hours 3 minutes. Un-

- Table 10. Per-corruption accuracy under Hard degradation, averaged over six MMD-Bench benchmarks. Corruptions are grouped by category. ∆ shows the improvement of CLEAR-RL over Bagel. Category averages are shown in italics.

Category Corruption Bagel CLEAR-RL ∆

Capture

lens blur 57.82 64.48 +6.66 lens flare 59.23 64.85 +5.62 motion blur 56.35 63.52 +7.17 dirty lens 58.94 64.25 +5.31 hsv saturation 59.51 64.63 +5.12 Category avg. 58.37 64.35 +5.98

Transmission

jpeg compression 61.18 66.12 +4.94 block exchange 58.73 64.31 +5.58 mean shift 61.82 66.93 +5.11 scan lines 60.14 65.62 +5.48 Category avg. 60.47 65.75 +5.28

Environmental

dark illumination 57.63 63.94 +6.31 atmospheric turbulence 58.35 64.12 +5.77 gaussian noise 59.84 66.25 +6.41 color diffusion 60.52 65.03 +4.51 Category avg. 59.09 64.84 +5.75

Post-proc.

sharpness change 62.31 67.08 +4.77 graffiti 59.84 63.42 +3.58 watermark damage 60.93 65.14 +4.21 Category avg. 61.03 65.21 +4.19

Overall 59.57 64.97 +5.40

- Table 11. CLEAR-RL evaluation time across degradation levels on each benchmark. R-Bench-Dis is evaluated only once as it contains pre-degraded images at a single severity level.

Benchmark High Mid Low MMBench 1h 35m 1h 01m 54m MM-Vet 9m 7m 7m MMVP 8m 6m 5m CV-Bench 31m 19m 15m MMStar 43m 31m 28m RealWorldQA 27m 17m 13m R-Bench-Dis 12m Total 3h 35m 2h 23m 2h 03m

der High degradation, the triggering rate rises to 36.4% and total time increases to 3 hours 35 minutes, a 74% increase. The per-benchmark pattern is consistent: MMBench, with the largest sample count (2,974), shows the largest absolute time increase (54m → 1h 35m), while smaller benchmarks like MMVP (300 samples) show proportionally smaller increases (5m → 8m).

The time difference between severity levels is entirely attributable to the adaptive generation policy. When generation is not triggered, the model performs only text reasoning with overhead comparable to Text-only CoT. When generation is triggered, the 30-step denoising process adds a fixed per-sample cost. The adaptive policy thus concentrates computational resources on inputs where generation yields the largest accuracy benefit, keeping overhead moderate under mild degradation while accepting the additional cost under severe degradation where the accuracy gains justify it.

### H. Hyperparameter Sensitivity

We analyze the sensitivity of CLEAR-RL to key hyperparameters by varying one parameter at a time while keeping all others at their default values. All experiments are evaluated on the six MMD-Bench benchmarks under both Clean and Hard degradation.

#### H.1. Flow-GRPO Loss Weight λ

The weight λ in LInterleaved = LGRPO+λLFlow-GRPO controls the relative contribution of the image generation objective.

- Table 12 reports the results across different values.

- Table 12. Effect of Flow-GRPO loss weight λ on accuracy (6bench average).

λ Clean Hard

- 0.0 69.85 63.52
- 0.1 70.06 64.31 0.3 (default) 70.27 64.97

- 0.5 70.18 64.72
- 1.0 69.71 63.89

Setting λ = 0 reduces Interleaved GRPO to text-only GRPO, where the generation process receives no direct optimization signal. This still outperforms SFT (63.04 Hard) because the text-side GRPO improves reasoning, but the generative pathway is not optimized and remains at its SFT initialization. Performance improves as λ increases from 0 to 0.3, confirming that coupling image generation to the reward signal is beneficial. Beyond 0.3, performance begins to decline: at λ = 1.0, the image-side gradients become too dominant relative to the text-side, slightly destabilizing the text reasoning process. The default value of 0.3 provides the best balance, and the method is reasonably robust within the range 0.1 to 0.5.

H.2. Reward Weights

- Table 13 examines the sensitivity to the three reward components by varying wacc, wfmt, and wdec.

- Table 13. Effect of reward weight configurations on accuracy (6bench average).

wacc wfmt wdec Clean Hard

1.0 0.0 0.0 69.92 64.18 0.85 0.1 0.05 70.13 64.55 0.75 0.1 0.15 (default) 70.27 64.97 0.65 0.1 0.25 70.08 64.61 0.60 0.15 0.25 69.83 64.24

Using only the accuracy reward (wdec = 0) yields a 0.79point drop on Hard compared to the default, because without the decision reward the model lacks a direct signal for

learning when to trigger generation. In this configuration the model tends to either over-generate on easy inputs or under-generate on hard inputs, as both behaviors can occasionally lead to correct answers and thus receive similar accuracy rewards. Increasing wdec to 0.15 provides the strongest performance by sharpening the generation decision. However, pushing wdec further to 0.25 shifts too much focus toward the binary generation decision at the expense of answer quality, leading to a slight decline. The format reward Rfmt is necessary to prevent degenerate outputs in early training but has limited influence on final performance as long as it is present.

#### H.3. Denoising Steps

Table 14 varies the number of denoising steps used during both training and inference.

Table 14. Effect of denoising steps on accuracy (6-bench Hard average) and per-sample denoising time (measured on samples that trigger generation).

Steps Hard Denoising Time

10 63.68 1.8s 20 64.53 3.5s 30 (default) 64.97 5.2s 50 65.04 8.7s

Accuracy improves from 10 to 30 steps as the denoising process has more iterations to recover fine details from the degraded input. The gain from 30 to 50 steps is marginal (0.07 points) while denoising time increases by 67%, making 30 steps a favorable trade-off between accuracy and efficiency. At 10 steps the denoising process is too coarse to recover the structural detail needed for reasoning, resulting in a 1.29-point drop compared to the default. The persample denoising time scales approximately linearly with the number of steps, confirming that the computational cost is predictable and controllable.

### I. Reward Design Details

This section provides additional details on the three reward components used in Interleaved GRPO.

#### I.1. Accuracy Reward Racc

The accuracy reward is computed by prompting GPT-4.1mini to compare the model’s answer with the ground-truth answer on a scale from 0 to 1. Figure 9 shows the prompt template. The judge is instructed to focus on semantic correctness rather than surface-level string matching, allowing partial credit for answers that capture the correct concept but differ in phrasing or formatting.

|LLM-as-Judge Prompt<br><br>You are an impartial judge evaluating the correctness of an AI assistant’s answer to a visual question.<br><br>Ground Truth: {ground truth} Model Answer: {model answer}<br><br>Rate the correctness of the model answer on a scale from 0.0 to 1.0, where 0.0 means completely wrong and 1.0 means perfectly correct. Focus on semantic meaning rather than exact wording. Award partial credit if the answer captures the correct concept but includes minor errors in phrasing, formatting, or specificity.<br><br>Output only a single number between 0.0 and 1.0.|
|---|

Figure 9. Prompt template for the LLM-as-judge accuracy evaluation.

#### I.2. Format Reward Rfmt

The format reward is a binary signal that checks whether the model output conforms to the expected structure. An output receives Rfmt = 1 if it contains properly formed <think> and <answer> blocks and follows one of the two valid patterns defined in the system prompt. Otherwise Rfmt = 0. This reward is intentionally simple, serving only to prevent degenerate outputs during early GRPO training.

#### I.3. Decision Reward Rdec

The decision reward evaluates the generation decision retrospectively based on the accuracy score. Let g ∈ {0,1} indicate whether the model triggered generation, and let c ∈ {0,1} indicate whether the answer is correct (defined as Racc > 0.5). The four cases are:

Table 15. Decision reward Rdec for the four possible outcomes.

Generated (g) Correct (c) Rdec Interpretation 1 1 +1 Generation helped

- 0 0 −1 Should have generated
- 1 0 0 Neutral 0 1 0 Neutral

The asymmetric design rewards successful generation and penalizes missed opportunities, while treating the remaining two cases as neutral. Incorrect answers after generation are not penalized because the model correctly identified the need for generation even though the restored information was insufficient. Correct answers without generation are not rewarded because no generation decision was needed. This design allows the adaptive generation policy to emerge naturally from the reward signal without requiring an explicit decision classifier.

### J. Qualitative Results

This section presents qualitative examples of CLEAR-RL’s reasoning behavior across different degradation types and

scenarios.

#### J.1. Reasoning Trace Examples

We first show complete reasoning trajectories to illustrate the two reasoning patterns and a representative failure case.

|Question: What text is written on the sign? Degradation: Motion blur (Hard)<br><br><think> The image suffers from severe motion blur. The sign in the center of the image has text on it, but the blur makes the characters unreadable. I cannot determine the text with confidence from the current image quality. I will request image restoration to recover the text details. </think><br><br><image restore><br><br><think> The restored image is much clearer. I can now see the sign in the center of the image. The text on the sign reads “NO PARKING”. The letters are white on a red background. </think><br><br><answer> The sign says “NO PARKING”. </answer>|
|---|

- Figure 10. Generate-then-answer trajectory. The model diagnoses motion blur, triggers restoration, and extracts the recovered text.

|Question: How many people are in the image? Degradation: Gaussian noise (Hard)<br><br><think> The image has noticeable Gaussian noise, but the overall scene is still discernible. I can see three people standing near a table in what appears to be a kitchen. The noise affects fine textures but the human figures are clearly distinguishable by their outlines and positioning. I can answer this question without restoration. </think><br><br><answer> There are three people in the image. </answer>|
|---|

- Figure 11. Direct-answer trajectory. The model judges that visual information is sufficient despite noise and answers without generation.

The failure case illustrates a limitation of the current approach. When the critical visual evidence occupies a very small region and is severely corrupted, the 30-step denoising process may not recover sufficient detail for correct identification, even though the overall image quality improves. This suggests that future work on region-aware or adaptive-resolution generation could further improve performance on such cases.

#### J.2. Additional Visual Examples

Figure 13 presents additional examples across different degradation types and benchmarks. Each example shows the degraded input, the intermediate visual state produced by CLEAR-RL (when generation is triggered), and the

|Question: What brand is the laptop on the desk? Degradation: JPEG compression (Hard)<br><br><think> The image is heavily compressed with visible blocking artifacts. I can see a laptop on a wooden desk, but the brand logo on the laptop lid is very small and the compression has destroyed the fine detail in that region. I will request restoration. </think><br><br><image restore><br><br><think> The restored image has improved overall clarity, but the brand logo remains too small and the compression artifacts in that specific region were too severe to fully recover. The shape of the logo suggests it could be Dell or HP, but I cannot determine this with certainty. Based on the overall shape, I will provide my best estimate. </think><br><br><answer> Dell </answer> Ground truth: HP|
|---|

Figure 12. Failure case. Generation improves overall quality but cannot recover the fine detail needed for the answer. The model identifies its uncertainty but guesses incorrectly.

model’s final answer alongside the ground truth. These examples further illustrate two key behaviors. First, the adaptive generation policy consistently triggers generation for severe degradations that obscure critical visual details while skipping generation for mild degradations where the understanding pathway alone is sufficient. Second, the intermediate visual states recover task-relevant structure such as text, object boundaries, and spatial layout, consistent with the finding that task-driven optimization prioritizes reasoning utility.

###### With Restoration Tool

###### Input (Degraded) Restored

[Figure 86]

[Figure 87]

###### (a)

Q: Roughly how much of the picture is occupied by the two people on the bench in the picture?

The image is heavily degraded with a mosaic or blocky effect, obscuring many parts of the scene. The relevant region for this question is th...

Invoke Restoration Tool

The restored image is clear, showing two people sitting on a bench facing a body of water. To answer the question, I need to visually estima...

A: B. less than 30%

[Figure 88]

[Figure 89]

###### (b)

Q: How many cars are in the opposite lane that are still driving?

The image is affected by significant noise, which makes it somewhat difficult to distinguish details, especially in the background where the...

Invoke Restoration Tool

The restored image is clear, and the opposite lane is visible in the background. The relevant evidence is the count of cars that are still d...

A: B. 2 cars

[Figure 90]

[Figure 91]

###### (c)

Q: Which traffic light is relevant to us?

The image is heavily overexposed, making it difficult to clearly see the details of the traffic lights. The relevant evidence needed is the ...

Invoke Restoration Tool

The restored image is clear and shows the traffic lights at the intersection. The relevant evidence is the color of the left arrow light and...

A: A. The red left arrow

[Figure 92]

[Figure 93]

###### (d)

###### Q: How many green trashcans are to my right?

The image is quite dark, making it difficult to clearly see the details on the right side of the street, where the trashcans are located. Th...

###### Invoke Restoration Tool

The restored image is clear and well-lit. To answer the question, I need to count the number of green trashcans visible to the right of the ...

###### A: B. 2

Figure 13. Additional qualitative examples across different degradation types.

