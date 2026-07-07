# arXiv:2505.15809v2[cs.CV]25Sep2025

[Figure 1]

[Figure 2]

## MMaDA: Multimodal Large Diffusion Language Models

### Ling Yang1,4∗†, Ye Tian2∗, Bowen Li2, Xinchen Zhang3, Ke Shen4, Yunhai Tong2, Mengdi Wang1

1Princeton University 2Peking University 3Tsinghua University 4ByteDance Seed ∗Equal Contribution

### Abstract

We introduce MMaDA, a novel class of multimodal diffusion foundation models designed to achieve superior performance across diverse domains such as textual reasoning, multimodal understanding, and text-to-image generation. The approach is distinguished by three key innovations: (i) MMaDA adopts a unified diffusion architecture with a shared probabilistic formulation and a modalityagnostic design, eliminating the need for modality-specific components. This architecture ensures seamless integration and processing across different data types. (ii) We implement a mixed long chain-of-thought (CoT) fine-tuning strategy that curates a unified CoT format across modalities. By aligning reasoning processes between textual and visual domains, this strategy facilitates coldstart training for the final reinforcement learning (RL) stage, thereby enhancing the model’s ability to handle complex tasks from the outset. (iii) We propose UniGRPO, a unified policy-gradientbased RL algorithm specifically tailored for diffusion foundation models. Utilizing diversified reward modeling, UniGRPO unifies post-training across both reasoning and generation tasks, ensuring consistent performance improvements. Experimental results demonstrate that MMaDA-8B exhibits strong generalization capabilities as a unified multimodal foundation model. It surpasses powerful models like LLaMA-3-7B and Qwen2-7B in textual reasoning, outperforms Show-o and SEED-X in multimodal understanding, and excels over SDXL and Janus in text-to-image generation. These achievements highlight MMaDA ’s effectiveness in bridging the gap between pretraining and post-training within unified diffusion architectures, providing a comprehensive framework for future research and development. We open-source our code and trained models at: https://github.com/Gen-Verse/MMaDA

Date: May 22, 2025 Correspondence: Ling Yang at yangling0818@163.com

### 1 Introduction

Large language models (LLMs) have revolutionized natural language processing (NLP) by achieving state-ofthe-art performance in diverse tasks, from text generation (e.g., ChatGPT [1–3]) to complex reasoning (e.g., DeepSeek-R1 [4]). Inspired by their success, the research community has extended LLMs to the multimodal domain, giving rise to multimodal large language models (MLLMs) or vision-language models (VLMs) [5–14], such as GPT-4 [15] and Gemini [12]. These models aim to provide a unified framework for both understanding and generating across heterogeneous modalities—text, images, and beyond.

Early multimodal approaches combined language models with diffusion models [16–19] to handle discrete (e.g., text) and continuous (e.g., image) modalities separately. Subsequent autoregressive (AR) methods simplified

- Table 1 Specific design choices employed by different unified multimodal foundation model families, including their

core loss functions. The next-token prediction loss is defined as LNTP = Exi [− log Pθ(xi | x<i)], representing the standard negative log-likelihood of generating the next token xi conditioned on its preceding context x<i. The training objective for continuous diffusion models is given by LDiff-cont = Et,x0∼q(x0),ϵ∼N(0,I),c ∥ϵ − ϵθ(xt, t, c)∥2 , where xt denotes the noised version of the original data x0 at timestep t, and c is an optional conditioning signal. The model learns to predict the noise ϵ added to the data, thereby enabling the reconstruction of x0 through an iterative denoising process defined as xt−1 = Fθ(xt, t). The discrete diffusion (masked token prediction) loss is given by LDiff-disc = Ez∗

−|M1 | i∈M log pθ(zi∗|z(maskedM) ) , which measures the average negative log-likelihood of correctly predicting the original discrete tokens zi∗ at positions masked by the set M, given the visible context provided by the masked sequence z(maskedM) .

i

###### AR

AR + Diffusion

AR + Diffusion

Ours MMaDA

(One Model)

(Two Models)

(One Model)

(One Model)

Network Architecture

Language AR AR AR Diffusion Vision AR Diffusion Diffusion Diffusion

Pre-training

Loss for Language LNTP LNTP LNTP LDiff-disc Loss for Vision LNTP LDiff-cont LDiff-cont or LDiff-disc LDiff-disc

Sampling

Language xi ∼ Pθ(xi | x<i) xi ∼ Pθ(xi | x<i) xi ∼ Pθ(xi | x<i) xt−1 = Fθ(xt,t) Vision xi ∼ Pθ(xi | x<i) xt−1 = Fθ(xt,t) xt−1 = Fθ(xt,t) xt−1 = Fθ(xt,t) Language Scheduler - - - Semi-AR Remask [22] Vision Scheduler - DDPM [23] DDPM [23]/Cosine [24] Cosine Remask [24]

Post-Training

Language CoT - - - Mixed Long-CoT Language RL - - - UniGRPO with Diversified RM Vision CoT - - - Mixed Long-CoT Vision RL DPO [25] - - UniGRPO with Diversified RM

Tasks

Und. (with Reasoning) ✓(✗) ✓(✗) ✓(✗) ✓(✓) Image Gen. (with Reasoning) ✓(✗) ✓(✗) ✓(✗) ✓(✓) Text Gen. (with Reasoning) ✗(✗) ✗(✗) ✗(✗) ✓(✓)

Representative Models Emu3 [25], Janus [13, 14] DreamLLM [16] Transfusion [21], Show-o [20] MMaDA (Ours)

architectures by training a single transformer with next-token prediction, unifying discrete and continuous generation in a single model [8–14]. Another line of work leverages modality-specific training objectives within a shared architecture: for example, Show-o [20] and Transfusion [21] combine autoregressive and diffusion modeling for modeling textual and visual semantics, respectively.

Although recent advancements have explored diffusion-based architectures for global context modeling and parallel generation [20, 21], existing unified multimodal foundation models predominantly focus on model architecture design and pretraining strategies, leaving a critical gap in the exploration of post-training methodologies, particularly in non-autoregressive settings.

To address this gap, we systematically investigate the design space for unified multimodal diffusion foundation models, introducing a novel framework that advances both architectural and training paradigms (a comprehensive comparison in table 1). This work bridges the gap between pretraining and post-training in unified multimodal diffusion models, offering a holistic framework for future research in this emerging field. Our contributions can be summarized as:

- • Unified Diffusion Foundation Architecture: We propose MMaDA, a class of diffusion-based models that extend traditional generators into generalist task solvers via a shared probabilistic formulation and modality-agnostic architecture. This design eliminates modality-specific components while maintaining strong performance across tasks.
- • Mixed Long-CoT Post-Training: We introduce mixed long chain-of-thought (CoT) finetuning to enable cold-start training. By curating a unified CoT format across tasks, we align reasoning processes between modalities (e.g., textual and visual), fostering cross-modal synergy and learning intermediate reasoning before final output generation.
- • Unified Reinforcement Learning (UniGRPO): We develop a unified diffusion-centric reinforcement learning

|Task 1: Textual Reasoning Question:<br><br>Given that<br><br>tanβ =<br><br>1<br>2<br><br><br>where β is an angle, determine the exact value of the following expression:<br><br>sin2 β − 3 sinβcosβ + 4 cos2 β|Answers from Other Models<br><br>Show-o, Emu3 and Janus Pro 7B do not support this task.|<think> To find the value of (sin2 β − 3sinβcosβ + 4cos2 β) given that tan β = 1/2, we can follow these steps:<br><br>1. **Express 𝒔𝒔𝒔𝒔𝒔𝒔 𝜷𝜷 and 𝒄𝒄𝒄𝒄𝒔𝒔𝜷𝜷 :** sinβ = tanβ/ 1 + tan2 β = 1/2 / 1 + 1/2 2 = 1/ 5<br><br>cosβ = 1/ 1 + tan2 β = 2/ 5<br><br>2. **Calculate each term:** sin2 β = 1/ 5 2 = 1/5<br><br>3sin βcosβ = 3 ⋅ 1/ 5 ⋅ 2/ 5 = 3 ⋅ 2/5 = 6/5 4cos2 β = 4 ⋅ 2/ 5 2 = 4 ⋅ 4/5 = 16/5<br><br>3. **Substitute and simplify:** sin2 β − 3sinβcosβ + 4cos2 β = 1/5 − 6/5 + 16/5 = 11/5<br><br><br>Final Answer: 𝟏𝟏𝟏𝟏/𝟓𝟓 </think> The exact value is 11/5.<br><br>Answer from MMaDA<br><br>[Figure 3]|
|---|---|---|
|Task 2: Multimodal Reasoning<br><br>[Figure 4]<br><br>Question:<br><br>In the diagram provided, lines AB and CD are parallel. The line EF intersects AB at point E and CD at point F. From point E, a perpendicular EG is drawn to EF, meeting CD at point G. Given that angle 1 is 52 degrees, what is the measure of angle 2?|Show-o:<br><br>Angle 2 measures 52 degrees.<br><br>Emu3:<br><br>Angle 2 is equal to angle 1 due to the parallel lines AB and CD. Since angle 1 measures 52 degrees, angle 2 also measures 52 degrees.<br><br>Janus Pro 7B:<br><br>Since line AB is parallel to line CD, corresponding angles are equal. Therefore, angle 1 and angle 2 are equal. Since angle 1 measures 52 degrees, angle 2 also measures 52 degrees. The answer is: 52<br><br>Answers from Other Models<br><br>[Figure 5]<br><br>[Figure 6]<br><br>[Figure 7]|Answer from MMaDA<br><br>[Figure 8]<br><br><think><br><br>To determine the measure of angle 2, we need to analyze the given information and use the properties of parallel lines and perpendicularity.<br><br>1. **Identify the given information:**<br><br>- Line AB is parallel to line CD.<br>- Points E and F are the intersections of line EF with lines AB<br><br><br>and CD respectively.<br><br>- Line EG is perpendicular to line EF, with point E as the foot of the perpendicular.<br><br>- Line EG intersects line CD at point G.<br>- Angle 1 measures 52 degrees.<br><br><br>2. **Understand the relationship between the angles:**<br><br>- Since EG is perpendicular to EF, angle FEG is 90 degrees.<br>- Therefore, angle 2 + angle 1 = 90 degrees.<br><br><br>3. **Determine the measure of angle 2:**<br><br>- Angle 2 = 90 degrees - 52 degrees = 38 degrees. So, the measure of angle 2 is 38 . </think> The measure of angle 2 is 38 degrees.<br><br>|
|Task 3: World Knowledge-Aware Text-to-Image Generation Prompt:<br><br>The most popular kind of animal among the people of Istanbul.|[Figure 9]<br><br>Images from Other Models<br><br>[Figure 10]<br><br>Show-o Emu3 Janus Pro 7B<br><br>[Figure 11]<br><br>[Figure 12]<br><br>[Figure 13]<br><br>[Figure 14]|CoT and Image from MMaDA<br><br><think> The description states that the most popular animal among the people of Istanbul is the cat. This is because cats are a popular animal for companionship and people in the city. </think> A close-up of a cat with a light brown and white fur pattern.<br><br>[Figure 15]<br><br>[Figure 16]|

Figure 1 Qualitative comparison across three tasks (more results in section C and section D.).

algorithm (UniGRPO) tailored for multimodal generation. This approach leverages diversified reward modeling to enhance the model’s ability to perform complex reasoning and maintain factual consistency in generation.

- • State-of-the-Art Performance: MMaDA achieves superior and balanced performance across three critical tasks: textual reasoning, multimodal understanding, and text-to-image generation. Notably, it outperforms both autoregressive and diffusion-based baselines in terms of accuracy, efficiency, and task adaptability.

### 2 MMaDA: Multimodal Large Diffusion Language Models

#### 2.1 Pretraining with Unified Diffusion Architecture and Objective

Data Tokenization To establish a unified modeling framework capable of processing both textual and visual data, we adopt a consistent discrete tokenization strategy across both modalities. This design enables the model to operate under a single modeling objective, i.e., the prediction of discrete masked tokens. For text tokenization, we utilize the tokenizer from LLaDA [22], which serves as the backbone for our MMaDA model. For image tokenization, we leverage the pretrained image quantizer adopted from Show-o [20], which is based on the MAGVIT-v2 [24] architecture and converts raw image pixels into sequences of discrete semantic tokens. Given an input image with dimensions H × W, the encoder generates a token map with dimensions Hf × Wf , where f represents the downsampling factor. In this implementation, we employ a downsampling factor of f = 16 and a codebook size of 8192. This configuration transforms a 512 × 512 pixel image into a sequence of 32 × 32 = 1024 discrete tokens. The transformed discrete image tokens are used in both understanding and

UniGRPO Diversified Rewards:

Mixed Long-CoT Data:

Three Diverse Tasks:

$#'()*+, % = '-∼/!(1|3)[ℱ(!#'( + ) − ./ + ]

Reasoningintensive tasks:

- 1

- 2 MultimodalUnderstanding

- 3 Visual Generation

Language Modeling

MixedLong-CoTData withUnifiedFormat

TrainingStage

Textual Reasoning

- 1

- 2

- 3

DeepSeek-R1 LMM-R1 …

[Figure 17]

###### !!": Correctness, Format, …

Primary CoTData

[Figure 18]

Multimodal Reasoning

World knowledgeaware T2I Gen:

###### !""#: Correctness, CLIP-Score, Format, …

Verify and Reformat

Text-to-Image Generation

[Figure 19]

GPT-4.1

!$%&: CLIP-Score, ImageReward, …

Stage1: Pretraining Stage2: Mixed Long-CoT Finetuning Stage3: UniGRPO Training

###### Uniformly Random Masking in Iterations

Discrete Diffusion Generation

###### Stage3:UniGRPOTraining

| | | | |
|---|---|---|---|

$;< (&,(=)

Iteration 1, p = 0.25

Iterative Denoising:

$;< (&,(%) $;< (&,(>)

Iteration 1, p = 0.50

| | |
|---|---|

Iteration 1, p = 0.75

Image/Text Prompt Image/Text Response

##### …

Remask Response for Masked Log Probs

A

1 ,

$;< =

$;< (&, (/)

Sequence Log Probs by Averaging

-

?@=

Masked Token Log Porbs:

Input Token Completion Token Mask Token Remask Token

1 |7|

0!" 1# 2, 3 =

8

91:3!(1#,(|2)

$!,#∈&

Textual Reasoning Multimodal Reasoning Text-to-Image Generation

[Figure 20]

Q: How many one-fourths are there in 7/2?

Q: What is the cat in the image doing?

Q: Generate an image about a famous landmark building in Europe.

[Figure 21]

[Figure 22]

[Figure 23]

###### InferenceStage

Text Tokenizer

Text Tokenizer

Text Tokenizer

###### Image Tokenizer

Reason and Generate with World Knowledge

MLLM (Discreate Diffusion)

A: <think> The Eiffel Tower is …</think> Eiffel Tower is reaching for the sunny sky.

A: <think> Okay, there’s a small white cat pushing… </think> The cat is pushing a small shopping cart.

[Figure 24]

A: <think> Asking this question is the same as … </think> 14

[Figure 25]

[Figure 26]

[Figure 27]

Figure 2 An overview of MMaDA pipeline.

generation modeling tasks.

Unified Probabilistic Formulation for Pretraining Recent unified multimodal frameworks aim to integrate multiple modeling objectives—such as autoregressive generation and diffusion-based denoising—into a single architecture for joint understanding and generation tasks [20, 21] (see preliminaries in section A.1). However, these approaches often introduce complex hybrid mechanisms that hinder model efficiency and coherence. In contrast, we propose a streamlined framework that not only simplifies the architectural complexity but also introduces a unified diffusion objective to model both visual and textual modalities under a shared probabilistic formulation. By aligning the noise corruption and semantic recovery processes across modalities, we enable more effective cross-modal interactions during pretraining, facilitating seamless integration of heterogeneous data sources.

Specifically, we formulate MMaDA as a mask token predictor (for both image and text tokens), a parametric model pθ(·|xt) that takes xt as input and predicts all masked tokens simultaneously. The model is trained with a unified cross-entropy loss computed only on the masked image/text tokens:

L

1 t

I[xit = [MASK]]log pθ(xi0|xt) , (1)

Lunify(θ) = −Et,x

0,xt

i=1

where x0 is ground truth, the timestep t is sampled uniformly from [0,1], and xt is obtained by applying the forward diffusion process to x0. I[·] denotes the indicator function to ensure that the loss is computed only over the masked tokens. Specific pretraining tasks are detailed in section 4.

#### 2.2 Post-Training with Mixed Long-CoT Finetuning

Cold Start Long-CoT Data Curation We investigate how CoT mechanisms [26] can enhance post-training for our unified multimodal diffusion framework and observe their effectiveness in promoting cross-modal synergies. To this end, we curate a compact dataset of long CoT trajectories across three core tasks: textual reasoning, multimodal reasoning, and text-to-image generation. This dataset enables stable post-training of our pretrained MMaDA model through the following principles:

- • Unified CoT Format: A critical challenge in vision-language foundation models is the heterogeneity of output formats across tasks (e.g., text vs. image generation). We propose a task-agnostic CoT format:

|<special_token>| < reasoning_process > |<special_token>| < result > .

The ‘<reasoning_process>‘ encodes step-by-step reasoning trajectories preceding the final output. This unified structure bridges modality-specific outputs and facilitates knowledge transfer between tasks. For instance, enhanced textual reasoning capabilities directly improve the realism of generated images by aligning semantic logic with visual synthesis.

- • Diversity, Complexity, and Accuracy: We leverage open-source large language and vision-language models (LLM/VLMs) to generate diverse reasoning trajectories across tasks (in figure 2). To ensure quality, we employ state-of-the-art models as verifiers to filter out inaccurate or shallow reasoning, selecting only high-quality, long-form CoT samples. Unlike prior unified models that focus on generic understanding and generation, our MMaDA is explicitly designed for:

- 1. Reasoning-intensive tasks (e.g., mathematical problem-solving), and
- 2. World-knowledge-aware text-to-image generation, where factual consistency is critical.

Mixed Long-CoT Finetuning Leveraging our unified diffusion architecture and probabilistic formulation, we develop a mixed-task long-CoT finetuning strategy to jointly optimize the model across heterogeneous tasks. This approach not only enhances task-specific capabilities but also creates a strong initialization for subsequent reinforcement learning (RL) stages. The training process follows these steps:

- 1. Prompt Preservation and Token Masking: We retain the original prompt p0 and independently mask tokens in the result (x0), denoted as rt.
- 2. Joint Input and Loss Computation: The concatenated input [p0,rt] is fed into our pre-trained mask predictor to compute the loss. This enables the model to reconstruct masked regions (r0) using contextual information from both the prompt and corrupted result.

The objective function is defined as:

LMixed-SFT = −Et,p

0,r0,rt

 1

t

L′

i=1

I[rti = [MASK]]log pθ(r0i|p0,rt)

 , (2)

where L′ denotes the sequence length. Here, [p0,r0] and [p0,rt] correspond to the clean data x0 and its noisy counterpart xt, respectively. This formulation ensures the model learns to recover masked tokens while maintaining alignment with the original prompt and task-specific reasoning logic.

- 2.3 Post-Training with Unified Reinforcement Learning

###### 2.3.1 Unified GRPO for Diffusion Foundation Models

With our mixed long-CoT fine-tuning, MMaDA demonstrates the ability to generate unified and coherent reasoning chains prior to final outputs. To further enhance its performance on knowledge-intensive tasks and complex reasoning/generation scenarios, we propose UniGRPO, a novel policy-gradient-based reinforcement learning algorithm tailored for diffusion foundation models. This approach enables a diffusion-centric RL training framework that unifies task-specific objectives across diverse modalities and reasoning paradigms. The method is structured into two core components: (1) a unified mathematical formulation for diffusion-based RL, and (2) diversified reward modeling to align policy gradients with task-specific rewards.

Challenges in Adapting Autoregressive GRPO to Diffusion Models The original GRPO [27] relies on computing token-level log-likelihoods πθ(oi,t|q,oi,<t) and sequence-level probabilities πθ and πref (preliminary in section A.2). In autoregressive (AR) LLMs, these metrics are efficiently derived via the chain rule of generation. However, diffusion models introduce three critical challenges:

- 1. (1) Local Masking Dependency: Token-level log-likelihoods log πθ(oi,t|q,oi,<t) are only valid within masked regions during the diffusion process, unlike AR models where all tokens are valid.
- 2. (2) Mask Ratio Sensitivity: A uniform mask ratio must be sampled for the response segment to approximate the policy distribution πθ, as diffusion dynamics depend on masking patterns.
- 3. (3) Non-Autoregressive Sequence-Level Likelihoods: The sequence-level log-likelihood cannot be directly accumulated from token-level probabilities due to the absence of an autoregressive chain rule in diffusion models.

Prior approaches address these issues with suboptimal strategies. LLaDA [22] employs Monte Carlo sampling over numerous mask ratios (e.g., 128 samples), incurring high computational costs for on-policy RL. d1 [28] fixes the mask ratio and randomizes question masking, which reduces noise diversity and ignores the multi-step denoising nature of diffusion.

Unified Formulation for Diffusion GRPO To overcome these limitations, we introduce UniGRPO, a computationally efficient approximation algorithm designed for diffusion architectures. Given a batch of responses {oi}Gi=1 for a query q, each response is fixed during gradient updates to ensure stable policy evaluation. We highlight three critical points of our UniGRPO as:

- 1. Structured Noising Strategy: For each oi, we sample a masking ratio pi ∈ [0,1] uniformly and construct a perturbed version o˜i,p by replacing tokens with [MASK]. The random seed for pi varies across gradient steps. This strategy aims to preserve stochasticity while ensuring the model is exposed to various stages of the diffusion denoising process, from nearly fully masked to nearly fully denoised answers. By doing so, UniGRPO learns from multi-step denoising information, which is consistent with conventional training methodologies for diffusion models and allows for the full utilization of their multi-step generative power.
- 2. Efficient Log-Likelihood Approximation: We define the expected per-token log-likelihood under the perturbed distribution as:

πθ′ (oi,t | q,o,p˜ i) = Ep

i∼[0,1] [I[oi,t,p = [MASK]]log pθ(oi,t,p|q)] (3) The sequence-level log-likelihood is then approximated by averaging over masked tokens:

πθ′ =

1 M o

i,t∈M

log pθ(oi,t|q), where M denotes the number of masked tokens. (4)

- 3. Policy Gradient Objective: The per-token reward is computed as the ratio between current and old

′ θ(oi,t|q,o,p˜ i)

policy likelihoods: ri,t′ (θ) = π

πold′ (oi,t|q,o,p˜ i). The final UniGRPO objective integrates clipped surrogate rewards and KL regularization:

JUniGRPO(θ) = E(q,a)∼D,{o

i}Gi=1∼πθold(·|q),{pi∈[0,1]}Gi=1

|oi|

G

1 |oi|

1 G

t=1

i=1

min ri,t′ (θ)Aˆi,t,

clip ri,t′ (θ),1 − ε,1 + ε A ˆi,t − βDKL(πθ′ ||πref′ ) ,

(5)

where Aˆi,t denotes the advantage estimate, ε controls the clipping range, and β balances the KL divergence penalty.

Through this design, UniGRPO captures the essential multi-step denoising dynamics of diffusion models. By allowing the model to predict answers under diverse masking conditions while preserving the natural

structure of the input, it avoids the pitfalls of both computational inefficiency (as in LLaDA) and oversimplified prediction (as in d1). The training procedure of UniGRPO is outlined in algorithm 1. For further details on the UniGRPO algorithm, please refer to section B and section 5.2.

Algorithm 1 UniGRPO Policy Gradient Optimization Require: Reference model πref, prompt distribution D, number of completions per prompt G, number of inner

updates µ, diffusion steps T

- 1: Initialize policy πθ ← πref
- 2: while not converged do
- 3: πold ← πθ
- 4: Sample a prompt q ∼ D
- 5: Sample G completions oi ∼ πold(· | q), for i ∈ [G]
- 6: For each oi, compute reward ri and advantage Aki (πold) using equation (15)
- 7: Sample a starting timestep t0 ∼ U(0,T − 1)
- 8: Generate µ − 1 uniformly spaced timesteps t1,...,tµ−1 from [t0,T]
- 9: for gradient update iterations n = 1,...,µ do
- 10: if n = 1 then
- 11: Sample a starting mask ratio r1 ∼ U(0,1) and compute initial timestep t1 = ⌊r1 · T⌋
- 12: else
- 13: Uniformly divide remaining timesteps: tn = ⌊((nµ−−1)1) · (T − t1) + t1⌋

- 14: Construct input (q,masked oi) using timestep tn (with q always unmasked)
- 15: For πθ, πold, πref, estimate log-probabilities of masked tokens in oi at tn
- 16: Compute UniGRPO objective equation (5) and update πθ via gradient descent
- 17: return πθ

- 2.3.2 Diversified Reward Modeling We further simplify the optimization objective of UniGRPO (equation (5)) as follows:

Remark 1. General optimization objective of UniGRPO:

JUniGRPO(θ) = Eo∼πθ(·|q)[F(RUni(o)) − βP(o)], (6)

where RUni(o) denotes the reward obtained from the model-generated response o, P(·) is the penalty term, which denotes the KL divergence as specified in equation (5). This is a unified rule-based reward system, where RUni(·) can be instantiated with diverse rewards for different tasks. To address the varied requirements of different tasks, we have defined a range of rewards under the unified formulation equation (6), providing tailored RL optimization directions for each task branch. We mainly adopt three types of rewards:

- • Textual Reasoning Rewards: We apply UniGRPO on the training split of the GSM8K [29] dataset and define a composite reward. This includes a Correctness Reward of 2.0 for a correct answer, and a Format Reward of 0.5 if the response adheres to our predefined format: “<think>...</think>”.
- • Multimodal Reasoning Rewards: For mathematical tasks such as GeoQA [30] and CLEVR [31], we adopt the same Correctness and Format Rewards as in textual reasoning. In addition, for caption-based tasks, we further introduce a CLIP Reward of 0.1 · CLIP(image,text), where the original CLIP score measuring text-image alignment is scaled by 0.1 to balance its influence.
- • Text-to-Image Generation Rewards: For image generation tasks, we incorporate the same CLIP Reward to assess text-image semantic alignment, alongside an Image Reward that reflects human preference scores. Both rewards are scaled by a factor of 0.1 to ensure balanced contribution during optimization.

### 3 Flexible Sampling Strategies at Inference Time

###### Semi-Autoregressive Sampling for Text Generation For text generation, we adopt the semi-autoregressive

denoising strategy introduced in LLaDA [22], which integrates autoregressive decoding with diffusion-based denoising. Specifically, the output sequence is partitioned into multiple blocks and generated from left to right. Within each block, logits are computed for all masked positions, and a subset of tokens is selected—either randomly or based on confidence scores—for denoising. The masking schedule follows a linear schedule, consistent with LLaDA. The denoising process is repeated for given steps.

In our evaluation, we set the total sequence length to N = 1024 and perform N2 = 512 denoising steps. The sequence is divided into blocks of 64 tokens. At each step, we unmask the 2 tokens with the lowest confidence

within the current block, irrespective of their positions. Once all tokens in a block are denoised, the process proceeds to the next block. A qualitative comparison is provided below. As shown, a semi-autoregressive denoising strategy tends to generate more intricate and detailed descriptions, whereas non-autoregressive fixed-length generation often produces very short responses. This observation is consistent with findings reported for LLaDA. For instruction-tuned models, given that the training process incorporates a substantial number of |EOS| tokens, directly applying the lowest-confidence remasking strategy without dividing into blocks leads to an unnaturally high frequency of |EOS| tokens in the generated sentences.

Qualitative Comparison of Different Sampling Strategy

[Figure 28]

Prompt: Please describe this image in detail.

Fix-length Generation ( Length = 1024, Steps = 512, Block Size = 1024)

The image is a detailed oil painting depicting a still life scene.

Semi-Autoregressive ( Length = 1024, Steps = 512, Block Size = 64)

The painting depicts a serene still life scene on a tablecloth with a textured, multicolored pattern. In the center, there is a rustic, cream-colored ceramic pitcher with brown accents, holding a vibrant bouquet of sunflowers. The sunflowers, with their bright yellow petals and deep brown centers, are arranged in a way that they seem to burst forth from the pitcher, creating a sense of abundance and vitality. To the right of the pitcher, there is a small, ornate glass vase with a patina finish, adding an element of contrast and visual interest to the composition. In the foreground, a sliced loaf of bread is placed on the tablecloth, with a piece of bread and a knife nearby, suggesting a meal or snack. The background is softly blurred, focusing attention on the main elements of the painting. The overall composition is balanced and harmonious, with the warm tones of the sunflowers and ceramic pitcher complementing the cooler bronze of the vase and the soft pastels of the bread.

Parallel Non-Autoregressive Sampling for Image Generation For image generation, we adopt a low-confidence remasking strategy and follow a cosine noise schedule, consistent with the setup in MAGVIT-v2 [24]. In contrast to text generation, we do not employ a semi-autoregressive approach; instead, the entire output sequence is treated as a single generation block. During evaluation, we generate sequences of length 1024, corresponding to 512×512 resolution images. The denoising process consists of 50 timesteps, and we apply classifier-free guidance with a guidance scale set to 3.5.

Table 2 Evaluation on Multimodal Understanding Benchmarks.

Model POPE↑ MME↑ Flickr30k↑ VQAv2(test)↑ GQA↑ MMMU ↑ MMB ↑ SEED ↑

Understanding-Only

LLaVA-v1.5 [32] 85.9 1510.7 - 78.5 62.0 35.4 64.3 58.6 InstructBLIP [33] 78.9 1212.8 - - 49.5 - - Qwen-VL-Chat [34] - 1487.5 - 78.2 57.5 - 60.6 58.2 mPLUG-Owl2 [35] 85.8 1450.2 - 79.4 56.1 - - LLaVA-Phi [36] 85.0 1335.1 - 71.4 - - 59.8 -

###### Unified Understanding & Generation

DreamLLM [16] - - - 72.9 - - - SEED-X [19] 84.2 1435.7 52.3 - 47.9 35.6 - Chameleon [8] - - 74.7 66.0 - - - LWM [9] 75.2 948.4 - 55.8 44.8 - - Emu [11] - - 77.4 57.2 - - - Show-o [20] 80.0 1097.2 62.5 69.4 58.0 26.7 - Gemini-Nano-1 [12] - - - 62.7 - 26.3 - MMaDA (Ours) 86.1 1410.7 67.6 76.7 61.3 30.2 68.5 64.2

### 4 Experiments

#### 4.1 Experimental Setup

Datasets To train MMaDA, we utilized a diverse range of datasets tailored for corresponding training stages as follows: (1) Foundational Language and Multimodal Data: For basic text generation capabilities, we adopt the RefinedWeb [37] dataset. For multimodal understanding and generation tasks, we incorporate widely-used open-sourced image-text datasets [38–42]. (2) Instruction Tuning Data: To enhance instructionfollowing capabilities, we use Alpaca [43] for textual instructions and LLaVA-1.5 [32] for visual instruction tuning. (3) Reasoning Data: For Mixed Long-CoT finetuning, we curated a diverse set of reasoning datasets. For textual mathematical and logical reasoning, we employed datasets from ReasonFlux [44], LIMO [45], s1k [46], OpenThoughts [47], and AceMath-Instruct [48]. For multimodal reasoning, we used the LMMR1 [49] model to generate responses on GeoQA [30] and CLEVR [31], and retained correctly answered instances. Additionally, for world knowledge-aware image generation, we used GPT-4.1 to synthesize factual item-description pairs spanning science, culture, and landmarks, formatted into unified CoT-style traces.(4) Reinforcement Learning Data: For UniGRPO training, we adopt the original mathematical and logical datasets used in Reasoning [30, 31, 50].

Evaluation and Baselines We evaluate our MMaDA on three distinct tasks using task-specific metrics and baselines:(1) Multimodal Understanding: Following LLaVA [32], we evaluate on POPE, MME, Flickr30k, VQAv2, GQA, and MMMU, and compare against understanding-only models [32–36], as well as unified models [8, 9, 11, 12, 16, 19, 20]. (2) Image Generation: We assess generation quality using 50K prompts from our test set to compute CLIP Score [51] and ImageReward [52] to evaluate textual alignment and human preference alignment. We adopt GenEval [53] for general evaluation and WISE [54] for evaluating world knowledge-based generation, comparing against generation-specific models [55–58] and unified baselines [8, 9, 11–

13, 16, 19, 20, 59]. (3) Text Generation: we evaluate instruction-following and reasoning performance on MMLU, GSM8K, and related benchmarks, comparing with LLaMA2-7B, Qwen2-7B, and LLaDA-8B.

Implementation Details We initialize MMaDA with LLaDA-8B-Instruct’s pretrained weights [22] and an image tokenizer with Show-o’s pretrained ones. We perform joint training across three stages: Stage1: The initial model is trained for 200K steps using foundational language and multimodal data, including RefinedWeb for text generation, ImageNet-1k for class-conditional image generation, and additional image-text datasets for

Table 3 Evaluation on Image Generation Benchmarks.

GenEval↑ Single Obj. Two Obj. Counting Colors Position Color Attr. Overall Generation-Only

↑ CLIP

Model Wise (Cultural)

↑ Image

↑

Score

Reward

LlamaGen [58] - 0.79 13.43 0.71 0.34 0.21 0.58 0.07 0.04 0.32 SDv1.5 [55] 0.34 0.84 23.54 0.97 0.38 0.35 0.76 0.04 0.06 0.43 SDv2.1 [55] 0.30 0.95 27.41 0.98 0.51 0.44 0.85 0.07 0.17 0.50 DALL-E 2 [56] - 0.83 25.20 0.94 0.66 0.49 0.77 0.10 0.19 0.52 SDXL [57] 0.43 1.13 32.12 0.98 0.74 0.39 0.85 0.15 0.23 0.55

###### Unified Understanding & Generation

DreamLLM [16] - 0.76 18.33 - - - - - - SEED-X [19] - 0.77 23.15 0.97 0.58 0.26 0.80 0.19 0.14 0.49 Chameleon [8] - 0.83 20.32 - - - - - - 0.39 LWM [9] - 0.78 26.21 0.93 0.41 0.46 0.79 0.09 0.15 0.47 Emu [11] - 0.81 22.29 - - - - - Show-o [20] 0.28 0.92 28.94 0.95 0.52 0.49 0.82 0.11 0.28 0.53 Janus [13] 0.16 1.03 29.45 0.97 0.68 0.30 0.84 0.46 0.42 0.61 Gemini-Nano-1 [12] - 0.89 24.58 - - - - - - VAR-GPT [59] - 0.94 28.85 0.96 0.53 0.48 0.83 0.13 0.21 0.53 MMaDA (Ours) 0.67 1.15 32.46 0.99 0.76 0.61 0.84 0.20 0.37 0.63

captioning. This is followed by another 400K steps where ImageNet is replaced with more diverse image-text pairs. Stage2: The model is then jointly trained for 50,000 steps using Instruction Tuning Data and Reasoning Data. Stage3: This final stage consists of UniGRPO training with Reinforcement Learning Data for 50,000 steps. Training is performed on 64 A100 (80GB) GPUs using a global batch size of 1,280. The AdamW optimizer is employed with an initial learning rate of 5e-5 and a cosine learning rate scheduler.

#### 4.2 Multimodal Understanding

- Table 2 reports the multimodal understanding performance of our method on standard benchmarks, including POPE, MME, Flickr30k, VQAv2, GQA, and MMMU. For outputs from MMaDA that contain reasoning traces, we use the final answer as the prediction. Compared with dedicated understanding-only models such as LLaVA-v1.5, InstructBLIP, and Qwen-VL-Chat, our model achieves comparable or superior results across most benchmarks, despite being trained under a unified objective. When compared to other unified models (e.g., SEED-X, DreamLLM, Janus, Emu3, and Show-o), our method consistently outperforms them across several benchmarks, particularly benefiting from the proposed Mixed Long-CoT Finetuning and UniGRPO Reinforcement Learning stages. Notably, this is the first demonstration of a diffusion-based MLLM exhibiting strong understanding capabilities, highlighting the potential of our unified architecture in bridging generation and understanding tasks. Qualitative results are in section C and section D.

- 4.3 Text-to-Image Generation

- Table 3 presents the evaluation results on text-to-image generation benchmarks. Our model achieves the highest performance in both CLIP Score and ImageReward across generation-only and unified models, attributed to the UniGRPO training stage with rewards explicitly aligned to these metrics. Furthermore, our method demonstrates superior compositionality and object counting capabilities on GenEval, benefiting from the reasoning-intensive training of the understanding branch. Notably, on WISE [54] Cultural benchmark, which is designed to evaluate world knowledge-aware generation, our model significantly outperforms prior approaches, owing to its joint training on text-based reasoning, which is typically absent in existing unified models. Qualitative results are in section C and section D.

Qualitative Comparison of Multimodal Reasoning

[Figure 29]

Question: In the diagram, triangle ABC is shown. Points D and E are located on segments AB and AC, respectively. Lines BE and CD intersect at point F. Given that angle BAC is 60°, angle DCA is 40°, and angle EBA is 30°, what is the measure of angle CFE?

Show-o: The angle measure of angle CFE is 60°.

Emu3: To solve for the degree measure of angle CFE, we can utilize the properties of triangles and angles. Since angle ABE measures 30°, angle ACD measures 40°, and angle ABC measures 60°, we can find angle ACD by subtracting angle A and angle ABE from 180°: 180° - 40° - 30° = 110°. Since triangle ABC is an isosceles triangle, angle BAC measures 40°. Therefore, angle CFE can be found by subtracting angle BAC from angle ABC: 40° - 110° = -30°. However, angles cannot have negative measures, so we take the positive value, giving us 60°. Therefore, the degree measure of angle CFE is 60°.

###### Janus Pro 7B:

To find the measure of angle CFE, we can use the properties of triangles and the given angles. First, we know that angle ABE is 30°. Since angle ABE and angle ACD are adjacent angles in the same triangle, they are supplementary. Therefore, angle ACD is 180° - 30° = 150°. Next, we know that angle A is 60°. Since angle A and angle ACD are adjacent angles in the same triangle, they are supplementary. Therefore, angle C is 180° - 60° = 120°. Now, we can find angle CFE. Angle CFE is the supplement of angle C, so angle CFE = 180° - 120° = 60°.

Therefore, the degree measure of angle CFE is 60°. MMaDA: <think> To determine the measure of angle CFE, we need to analyze the given information and use the properties of triangles and angles.

- 1. **Identify the given angles:**

- - Angle A = 60°
- - Angle ACD = 40°
- - Angle ABE = 30°

- 2. **Use the fact that the sum of the angles in a triangle is 180°:**

- In a triangle, the sum of the angles is 180°. Therefore, we have: 60◦ + ∠ADC + 40◦ = 180◦ 60◦ + ∠AEB + 30◦ = 180◦

- 3. **Determine the measure of angle CFE:**

- - Since angle AEB is an exterior angle to triangle CEF, it is equal to the sum of the two non-adjacent interior angles. Therefore, angle AEB = angle ACD + angle CFE = 40° + angle CFE = 90°.
- - Therefore, angle CFE = 90° - 40° = 50°.

|50|
|---|

.

Thus, the measure of angle CFE is

###### </think>

|50|
|---|

The measure of angle CFE is

.

Qualitative Comparison of World Knowledge-Aware Text-to-Image Generation Prompt: A famous statue gifted to the United States by France. Other models:

Show-o Emu3 Janus Pro 7B

[Figure 30]

[Figure 31]

[Figure 32]

MMaDA: <think>

The description refers to a well-known landmark that was given from France to the United States. This is known as the Statue of Liberty.

</think> A low-angle shot of the Statue of Liberty against a blue sky with a few wispy clouds.

[Figure 33]

#### 4.4 Textual Reasoning

###### Table 4 Evaluation on LLM Benchmarks.

Model Arch MMLU ARC-C TruthfulQA GSM8K MATH GPQA

- LLaMA-2-7B AR 45.9 46.3 39.0 14.3 3.2 25.7
- LLaMA-3-8B AR 64.5 53.1 44.0 53.1 15.1 25.9 Qwen2-7B AR 70.3 60.6 54.2 80.2 43.5 30.8

LLaDA-8B Diffusion 65.9 47.9 46.4 70.7 27.3 26.1 MMaDA-8B(Ours) Diffusion 68.4 57.4 43.1 73.4 36.0 28.4

- table 4 details the language modeling performance of MMaDA across a range of benchmarks, encompassing general tasks such as MMLU, ARC-C, and TruthfulQA, as well as mathematical tasks including GSM8K, MATH, and GPQA. Despite being trained on limited task-specific tokens and solely open-source text data, MMaDA achieves comparable performance compared to strong baselines such as Qwen2-7B and LLaMA3-8B on MMLU, ARC-C, and consistently outperforms LLaDA-8B on math benchmarks. Notably, MMaDA (Ours) pioneers the joint training of a unified diffusion model for text generation, multimodal reasoning, and image generation—a multi-task configuration rarely explored in prior unified architectures. These results underscore the viability of diffusion-based models as general-purpose LLMs and indicate potential for stronger future performance through enhanced text data and scaling. Qualitative results are in section C.

###### Qualitative Comparison of Textual Reasoning

Question: What is the result of (20−16)4×(12+8)? Show-o:

< Show-o does not support this task >

Emu3:

< Emu3 does not support this task >

Janus Pro 7B:

< Janus Pro 7B does not support this task >

MMaDA: <think>

Okay, let me try to figure out the result of this expression: (20−16)4×(12+8) . Hmm, let’s break this down step by step. First, I need to make sure I apply the order of operations correctly. Right, order of operations: parentheses

first, then multiplication and division from left to right, then addition and subtraction from left to right. Let me write the expression again to visualize it better: (20−16)4×(12+8) . So, the parentheses are within the numerator, which is part of the fraction, and the denominator is 4. According to PEMDAS, I should handle the parentheses first before doing the multiplication and division. Let me look at the parentheses part: (20 - 16) and (12 + 8). Both of those are inside parentheses, so I should compute each part separately. Starting with the first parentheses: 20 - 16. That’s straightforward. 20 minus 16 equals 4. Okay, so that part simplifies to 4. Next, the second part inside the parentheses: 12 + 8. That’s addition, so 12 plus 8 is 20. So now, the numerator becomes 4 × 20. So now, the numerator is 4 × 20. Let me do that multiplication. 4 times 20 is 80. So the numerator simplifies to 80. Now the entire expression is 804 . Dividing that number by 4. Let me check that division. 4 goes into 80 twenty times (since 4 × 20 is 80), so 80 ÷ 4 = 20. So the result is 20. Wait a second, let me verify that again to make sure I didn’t make a mistake. Let’s go through each step again. Original expression: (20−16)4×(12+8) . First parentheses: 20 - 16 = 4. Check. Second parentheses: 12 + 8 = 20. Check. Multiply those two: 4 × 20 = 80. Check. Divide by 4: 80 ÷ 4 = 20. Yep, that seems right. But wait, maybe there’s another way to approach this. Sometimes with order of operations, parentheses are easier to catch. Let’s see. The original expression is (20−16)4×(12+8) . The parentheses in the numerator give us

multiplication, and the denominator is 4. You could write it as 4×420 . Simplifying, 4 × 20 = 80, then 80 divided by 4 is 20. Alternatively, you could factor out the 4 in the numerator and cancel with the denominator, but in this case,

the result is still 20. So that’s a good way. Either way, multiplication and division are done from left to right. Alternatively, let’s check if there’s any possibility that the entire numerator is being divided. Let’s see. The

entire expression is 4×420 , so 4 × 20 is 80, then 80 divided by 4 is 20. So that’s another way to see it, which is reassuring. So that seems consistent.

Another way to check: look at it by using properties of operations. But in this case, since the expression is written with multiplication and division, the order of operations tells us to do all calculations in the numerator first, then divide by the denominator. All steps lead to 20. Another check: let me verify the multiplication. 4 × 20 = 80. Then 80 divided by 4 is 20. So no errors in the operations. Hmm, I think that’s solid. The answer should be 20. Let me just make sure there’s no alternative interpretation. Let’s write the original expression again:

Original expression: (20−16)4×(12+8) . The numerator is 4 × 20 = 80, denominator is 4. So 80 divided by 4 is 20. Yep, exactly, that’s correct.

Therefore, I’m confident that the answer is 20.

###### </think>

The result of (20−16)4×(12+8) is calculated as follows:

- 1. **Simplify the parentheses inside the numerator:**

- - 20 − 16 = 4
- - 12 + 8 = 20

- 2. **Multiply the simplified results:**

- 4 × 20 = 80

- 3. **Divide by 4:**

- 804 = 20

|20|
|---|

**Final Answer:**

### 5 Observations, Analysis and Conclusion

#### 5.1 Ablation Study

We present quantitative ablation results of our MMaDA across different training stages: Mixed Long-CoT fine-tuning and UniGRPO. All results generated follow the sampling process in section 3. As shown in the

- table 5, after Stage 1, our model still lags behind most baselines. In Stage 2, Mixed Long-CoT fine-tuning substantially enhances the model’s reasoning capabilities, particularly in mathematical and geometric domains. In Stage 3, UniGRPO further improves performance, allowing the model to achieve results comparable to state-of-the-art methods across various tasks, including mathematical reasoning, geometric problem-solving, and image generation benchmarks such as CLIP Score and ImageReward. These results demonstrate that UniGRPO effectively boosts both the model’s understanding/reasoning and generative capabilities.

#### 5.2 Design Choices of UniGRPO

Effect of General Masking Strategy To evaluate the impact of our proposed masking strategy, we first conduct a comparative analysis with d1 [28] within our reinforcement learning framework. Given the substantial computational cost associated with large-scale ablation studies, we perform these experiments on the GSM8K dataset, utilizing 8 A100 GPUs. Both the original d1 methodology and our UniGRPO approach are applied to this dataset, starting from the same pre-trained checkpoint of our MMaDA. We present the reward trends during training in figure 3. As shown in the figure, our method consistently achieves higher reward values during training, aligning well with our theoretical analysis. In contrast to d1, UniGRPO removes masking from the question and applies partial masking to the answer rather than masking it entirely. This results in input sequences that retain partial noise, encouraging the model to learn across multiple denoising timesteps. Consequently, this better leverages the intrinsic characteristics of diffusion models and improves the overall learning capacity.

UniGRPO

2.5

diff-GRPO

CorrectnessReward

2.0

1.5

1.0

0.5

0 1000 2000 3000 4000 5000

RL Steps

Figure 3 Comparison of different masking strategies on GSM8K reward trends during training.

Effect of Uniformly Random masking In place of fully random masking across iterations, we adopt a uniformly random masking strategy for the answer portion. Specifically, we first sample a random starting timestep, and then uniformly generate the remaining denoising timesteps across the full diffusion timesteps (set to 1000 in our experiments). For instance, given a randomly selected starting timestep of 100 and a total of 5 training iterations, the remaining timesteps are uniformly spaced and set to 300, 500, 700, and 900. This design ensures a more consistent coverage of the diffusion process while retaining randomness. We illustrate the training reward trends resulting from this structured masking strategy in figure 4. As shown, the baseline approach with fully random timestep selection tends to introduce instability during training, leading to more frequent reward fluctuations and requiring a greater number of steps to converge. In contrast, our uniformly spaced sampling strategy effectively approximates the behavior of Monte Carlo averaging in log-likelihood estimation, resulting in improved stability and faster convergence.

2.5

CorrectnessReward

2.0

1.5

1.0

0.5

Uniformly Random

Fully Random

0 1000 2000 3000 4000 5000

RL Steps

Figure 4 Comparison of different random masking strategies on GSM8K reward trends during training.

Text MMU Gen

[Figure 34]

Q: Give a detailed description of image

###### Q: What does a greenhouse look like?

###### Prompt: a large greenhouse filled with various plants

[Figure 35]

A close-up of a textured surface with a mix of green and brown hues, suggesting a blend of natural and artificial elements. the texture is intricate, with a variety of shapes and lines that create a sense of depth and dimensionality. the background is abstract, with a multitude of shapes and lines that give the impression of a complex, organic environment.

Steps 10K

A: A greenhouse is a covered structure used to grow plants, flowers, and crops.

[Figure 36]

A: The image features a large greenhouse filled with many types of plants. The greenhouse is covered with a clear roof, which allows for ample natural light to enter the space. The plants are arranged in rows, creating a greenhouse-like environment. The style of the image is a close-up view, focusing on the plants and the greenhouse's structure. This style emphasizes the beauty and diversity of the plants and the greenhouse's design.

A: A greenhouse is a long, narrow structure designed to grow plants. It is usually covered with a layer of plastic or made from fabric, which provides a controlled environment for plants and protects them from being made too exposed to the elements. Greenhouses can be used for a wide variety of fruits, vegetables, and other plants

Steps 100K

Figure 5 Qualitative Illustration of Synergy Across Modalities.

Table 5 Ablations on Mixed Long-CoT fine-tuning and UniGRPO

Model GSM8K MATH500 GeoQA CLEVR CLIP Score ImageReward

MMaDA After Stage 1 17.4 4.2 8.3 10.3 23.1 0.69 + Mixed Long-CoT Finetuning 65.2 26.5 15.9 27.5 29.4 0.84 + UniGRPO (MMaDA) 73.4 36.0 21.0 34.5 32.5 1.15

#### 5.3 Synergy Across Various Tasks

Throughout the joint training process, we observe a clear synergy across the three task categories—text generation, multimodal understanding, and image generation. As shown in figure 6, all key performance metrics exhibit consistent improvements during Stage 2 (training steps 120K–200K), reflecting the mutually beneficial nature of our unified training framework. This synergy is also evident qualitatively: as illustrated in figure 5, the model’s responses—both textual and visual—become increasingly complex and coherent. Specifically, textual outputs grow more informative and logically structured, while visual understanding yields more precise and grounded descriptions. Consequently, for the same prompt, the generated images become more accurate, detailed, and better aligned with the given instructions, demonstrating the effectiveness of joint optimization in enhancing cross-modal alignment and compositionality.

Synergy Across Tasks During Joint Training

0.70

| |
|---|

| |
|---|

NormalizedPerformance

0.65

| |
|---|

0.60

| |
|---|

| |
|---|

0.55

0.50

Text Generation (MMLU)

Multimodal Understanding (CLIP Score)

0.45

Image Generation (Image Reward)

130K 200K Training Steps

Figure 6 Key Performance Metrics Across Three Tasks.

#### 5.4 Sampling Efficiency

We identify sampling efficiency as a key advantage of diffusion models over autoregressive (AR) approaches. Unlike AR models, which generate tokens sequentially, diffusion models enable parallel token generation within each denoising step, substantially reducing the number of forward passes required. To quantify this advantage, we evaluate the performance of MMaDA under varying numbers of denoising steps. In our setup, generation begins from 1024 [MASK] tokens, allowing up to 1024 denoising steps—corresponding to a 512×512 resolution image. As shown in table 6, image generation maintains strong performance even with as few as 15 or 50

Table 6 Generation performance of MMaDA under different denoising steps. * Metrics: CLIP Score for image generation and multimodal understanding, MMLU accuracy for text generation.

Task Denoising Steps Metrics*

1024 32.8 50 32.0 15 31.7

Image Generation

1024 35.5 512 36.1 256 35.4

Multimodal Understanding

1024 66.9 512 66.3 256 65.7

Text Generation

steps. For text and multimodal tasks, coherent outputs can be achieved with just a quarter or half of the full steps. These results underscore the efficiency potential of diffusion-based language models and suggest that future advances in sampling techniques or higher-order solvers could further enhance their speed and quality.

#### 5.5 Task Extension

A notable advantage of diffusion-based models is their natural ability to perform inpainting and extrapolation without requiring additional fine-tuning. This stems from the fact that these tasks can be formulated as masked token prediction problems, which are inherently integrated into the training objective of diffusion models. While prior work such

The capital city of France is Paris. It is known for the Eiffel Tower and its rich cultural heritage.

The capital city of France is Paris. It is known for the Eiffel Tower and its rich cultural heritage.

TextUndGen

[Figure 37]

The image features a beautiful blonde woman standing in a field of white flowers.

The image features a beautiful blonde woman standing in a field of white flowers.

- as Show-o demonstrates this property only in the context of image generation, MMaDA extends it further to multimodal understanding and text generation. As illustrated in Figure 7, our model supports inpainting across three modalities: (i) predicting missing spans in text sequences, (ii) completing answers in visual question answering given an image and partial input, and (iii) performing image inpainting conditioned on incomplete visual prompts. These examples showcase the flexibility and generalization capabilities of our unified diffusion architecture across diverse generation and reasoning tasks.

[Figure 38]

[Figure 39]

A yellow racing car with sleek curves and tinted windows, parked on a bustling city street.

Figure 7 Inpainting Task Extension.

### 6 Related Work

Multimodal Large Language Models for Multimodal Understanding Recent developments in large language models (LLMs) such as Gemini-2.0 [60], o1-preview [61], and DeepSeek-R1 [62] have improved the evolution of multimodal large language models (MLLMs) [63–65]. Early efforts in this domain, including LLaVA [66], MiniGPT-4 [67], and InstructBLIP [33], showcased impressive capabilities in multimodal understanding. These studies advanced the integration of LLMs into multimodal contexts by projecting features from pretrained modality-specific encoders, such as CLIP [51], into the input space of LLMs, thereby facilitating multimodal understanding and reasoning within a unified transformer. Many efforts have been made for MLLMs regarding vision encoders, alignment adapters, and curated datasets [34, 68–70], and most of them follow an autoregressive generation paradigm that has been proven effective for text generation in LLMs. However, they are usually not capable of performing textual and multimodal reasoning concurrently. In this work, MMaDA develops diffusion foundation models to fill this gap.

Diffusion Models and Autoregressive Models for Visual Generation A large number of diffusion models [55–57, 71–85] have demonstrated notable success in visual generation. In addition to the typical denoising diffusion process on the continuous space, a series of frameworks, such as D3PM [86] and VQDiffusion [87], adopt discrete diffusion modeling [86, 88] for visual generation. Specifically, the image is denoted as a sequence of discrete tokens using pretrained image tokenizers [24, 89–91]. In tthe raining stage, the model is optimized to recover the original values of a portion of these tokens that are randomly masked. Transformer series [1, 2, 92–94] has demonstrated significant capabilities in autoregressive modeling for NLP tasks. Many approaches [95–99] try to apply the autoregressive modeling to perform visual generation by modeling semantic dependency within visual details. For example, LlamaGen [58] employs Llama architectures [94] and refines codebook design to enhance the performance of discrete tokenizers in class-conditional image generation. VAR [100] replaces the “next-token prediction” paradigm with “next-scale prediction” by designing a multi-scale image tokenizer. However, existing autoregressive methods still lag behind diffusion methods in terms of visual generation capabilities. In this work, MMaDA train diffusion models to model textual and visual contents, and infer efficiently with AR or Semi-AR sampling.

Unified Vision-Language Foundation Models Recently, numerous studies [16, 19, 101–104] have focused on developing unified multimodal foundation models that excel in both understanding and generation. Approaches such as SEED-X [19] and DreamLLM [16], along with others [5–14], represent all modalities as a series of tokens and employ a unified transformer architecture to train the entire system end-to-end. For example, Emu3 [25] trains a single transformer from scratch using a mixture of multimodal tokenized sequences, optimized solely with next-token prediction. While these unified autoregressive models show promise, they can struggle with visual generation tasks. Transfusion [21] and Show-o [20] employ autoregressive modeling for text generation and diffusion modeling for visual generation. Nevertheless, they mainly focus on pretraining strategies. Exploring effective post-training designs is still lacking for existing unified multimodal foundation models.

### 7 Conclusion

This work introduces a unified diffusion foundation model, namely MMaDA, that integrates textual reasoning, multimodal understanding, and generation within a single probabilistic framework. To the best of our knowledge, MMaDA is the first to systematically explore the design space of diffusion-based foundation models, proposing novel post-training strategies. Extensive experiments across diverse vision-language tasks demonstrate that MMaDA is comparable to or even better than specialized models, highlighting the potential of diffusion models as a next-generation foundation paradigm for multimodal intelligence. However, MMaDA also has limitations due to its current model size (8B parameters), and we will use a larger model size for better performance.

### References

- [1] Alec Radford, Karthik Narasimhan, Tim Salimans, Ilya Sutskever, et al. Improving language understanding by generative pre-training. 2018.
- [2] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel Ziegler, Jeffrey Wu, Clemens Winter, Chris Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners. In Advances in Neural Information Processing Systems, volume 33, 2020.

- [3] Aaron Jaech, Adam Kalai, Adam Lerer, Adam Richardson, Ahmed El-Kishky, Aiden Low, Alec Helyar, Aleksander Madry, Alex Beutel, Alex Carney, et al. Openai o1 system card. arXiv preprint arXiv:2412.16720, 2024.

- [4] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.

- [5] Jinguo Zhu, Xiaohan Ding, Yixiao Ge, Yuying Ge, Sijie Zhao, Hengshuang Zhao, Xiaohua Wang, and Ying Shan. VL-GPT: A generative pre-trained transformer for vision and language understanding and generation. CoRR, abs/2312.09251, 2023.

- [6] Quan Sun, Qiying Yu, Yufeng Cui, Fan Zhang, Xiaosong Zhang, Yueze Wang, Hongcheng Gao, Jingjing Liu, Tiejun Huang, and Xinlong Wang. Generative pretraining in multimodality. CoRR, abs/2307.05222, 2023.

- [7] Quan Sun, Yufeng Cui, Xiaosong Zhang, Fan Zhang, Qiying Yu, Zhengxiong Luo, Yueze Wang, Yongming Rao, Jingjing Liu, Tiejun Huang, and Xinlong Wang. Generative multimodal models are in-context learners. CoRR, abs/2312.13286, 2023.

- [8] Chameleon Team. Chameleon: Mixed-modal early-fusion foundation models. arXiv preprint arXiv:2405.09818, 2024.

- [9] Hao Liu, Wilson Yan, Matei Zaharia, and Pieter Abbeel. World model on million-length video and language with ringattention. arXiv preprint, 2024.

- [10] Yecheng Wu, Zhuoyang Zhang, Junyu Chen, Haotian Tang, Dacheng Li, Yunhao Fang, Ligeng Zhu, Enze Xie, Hongxu Yin, Li Yi, et al. Vila-u: a unified foundation model integrating visual understanding and generation. arXiv preprint arXiv:2409.04429, 2024.

- [11] Quan Sun, Qiying Yu, Yufeng Cui, Fan Zhang, Xiaosong Zhang, Yueze Wang, Hongcheng Gao, Jingjing Liu, Tiejun Huang, and Xinlong Wang. Emu: Generative pretraining in multimodality. In ICLR, 2023.

- [12] Rohan Anil, Sebastian Borgeaud, Yonghui Wu, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, Katie Millican, et al. Gemini: A family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 1, 2023.

- [13] Chengyue Wu, Xiaokang Chen, Zhiyu Wu, Yiyang Ma, Xingchao Liu, Zizheng Pan, Wen Liu, Zhenda Xie, Xingkai Yu, Chong Ruan, et al. Janus: Decoupling visual encoding for unified multimodal understanding and generation. arXiv preprint arXiv:2410.13848, 2024.

- [14] Xiaokang Chen, Zhiyu Wu, Xingchao Liu, Zizheng Pan, Wen Liu, Zhenda Xie, Xingkai Yu, and Chong Ruan. Janus-pro: Unified multimodal understanding and generation with data and model scaling. arXiv preprint arXiv:2501.17811, 2025.

- [15] OpenAI. Gpt-4 technical report. 2023.
- [16] Runpei Dong, Chunrui Han, Yuang Peng, Zekun Qi, Zheng Ge, Jinrong Yang, Liang Zhao, Jianjian Sun, Hongyu Zhou, Haoran Wei, Xiangwen Kong, Xiangyu Zhang, Kaisheng Ma, and Li Yi. DreamLLM: Synergistic multimodal comprehension and creation. In ICLR, 2024.

- [17] Jing Yu Koh, Daniel Fried, and Russ R Salakhutdinov. Generating images with multimodal language models. Advances in Neural Information Processing Systems, 36:21487–21506, 2023.

- [18] Shilong Liu, Hao Cheng, Haotian Liu, Hao Zhang, Feng Li, Tianhe Ren, Xueyan Zou, Jianwei Yang, Hang Su, Jun Zhu, et al. Llava-plus: Learning to use tools for creating multimodal agents. In European Conference on Computer Vision, pages 126–142. Springer, 2024.

- [19] Yuying Ge, Sijie Zhao, Jinguo Zhu, Yixiao Ge, Kun Yi, Lin Song, Chen Li, Xiaohan Ding, and Ying Shan. Seed-x: Multimodal models with unified multi-granularity comprehension and generation. arXiv preprint arXiv:2404.14396, 2024.

- [20] Jinheng Xie, Weijia Mao, Zechen Bai, David Junhao Zhang, Weihao Wang, Kevin Qinghong Lin, Yuchao Gu, Zhijie Chen, Zhenheng Yang, and Mike Zheng Shou. Show-o: One single transformer to unify multimodal understanding and generation. arXiv preprint arXiv:2408.12528, 2024.

- [21] Chunting Zhou, Lili Yu, Arun Babu, Kushal Tirumala, Michihiro Yasunaga, Leonid Shamis, Jacob Kahn, Xuezhe Ma, Luke Zettlemoyer, and Omer Levy. Transfusion: Predict the next token and diffuse images with one multi-modal model. arXiv preprint arXiv:2408.11039, 2024.

- [22] Shen Nie, Fengqi Zhu, Zebin You, Xiaolu Zhang, Jingyang Ou, Jun Hu, Jun Zhou, Yankai Lin, Ji-Rong Wen, and Chongxuan Li. Large language diffusion models. arXiv preprint arXiv:2502.09992, 2025.

- [23] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In NeurIPS, pages 6840–6851, 2020.

- [24] Lijun Yu, José Lezama, Nitesh B Gundavarapu, Luca Versari, Kihyuk Sohn, David Minnen, Yong Cheng, Agrim Gupta, Xiuye Gu, Alexander G Hauptmann, et al. Language model beats diffusion–tokenizer is key to visual generation. arXiv preprint arXiv:2310.05737, 2023.

- [25] Xinlong Wang, Xiaosong Zhang, Zhengxiong Luo, Quan Sun, Yufeng Cui, Jinsheng Wang, Fan Zhang, Yueze Wang, Zhen Li, Qiying Yu, et al. Emu3: Next-token prediction is all you need. arXiv preprint arXiv:2409.18869, 2024.

- [26] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824–24837, 2022.

- [27] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Y Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

- [28] Siyan Zhao, Devaansh Gupta, Qinqing Zheng, and Aditya Grover. d1: Scaling reasoning in diffusion large language models via reinforcement learning, 2025.
- [29] Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.

- [30] Jiaqi Chen, Jianheng Tang, Jinghui Qin, Xiaodan Liang, Lingbo Liu, Eric P. Xing, and Liang Lin. Geoqa: A geometric question answering benchmark towards multimodal numerical reasoning, 2022.
- [31] Justin Johnson, Bharath Hariharan, Laurens van der Maaten, Li Fei-Fei, C. Lawrence Zitnick, and Ross Girshick. Clevr: A diagnostic dataset for compositional language and elementary visual reasoning, 2016.
- [32] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. In CVPR, pages 26296–26306, 2024.

- [33] Wenliang Dai, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale Fung, and Steven Hoi. Instructblip: Towards general-purpose vision-language models with instruction tuning, 2023.
- [34] Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A frontier large vision-language model with versatile abilities. CoRR, abs/2308.12966, 2023.

- [35] Qinghao Ye, Haiyang Xu, Jiabo Ye, Ming Yan, Anwen Hu, Haowei Liu, Qi Qian, Ji Zhang, and Fei Huang. mplug-owl2: Revolutionizing multi-modal large language model with modality collaboration. In CVPR, pages 13040–13051, 2024.

- [36] Yichen Zhu, Minjie Zhu, Ning Liu, Zhiyuan Xu, and Yaxin Peng. Llava-phi: Efficient multi-modal assistant with small language model. In Proceedings of the 1st International Workshop on Efficient Multimedia Computing under Limited, pages 18–22, 2024.

- [37] Guilherme Penedo, Quentin Malartic, Daniel Hesslow, Ruxandra Cojocaru, Hamza Alobeidli, Alessandro Cappelli, Baptiste Pannier, Ebtesam Almazrouei, and Julien Launay. The refinedweb dataset for falcon LLM: outperforming curated corpora with web data only. In NeurIPS, 2023.

- [38] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In CVPR, pages 248–255, 2009.

- [39] Soravit Changpinyo, Piyush Sharma, Nan Ding, and Radu Soricut. Conceptual 12m: Pushing web-scale image-text pre-training to recognize long-tail visual concepts. In CVPR, pages 3558–3568, 2021.

- [40] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. In ICCV, pages 4015–4026, 2023.

- [41] David McClure. laion-aesthetics-12m-umap. https://huggingface.co/datasets/dclure/ laion-aesthetics-12m-umap, 2024.

- [42] Keqiang Sun, Junting Pan, Yuying Ge, Hao Li, Haodong Duan, Xiaoshi Wu, Renrui Zhang, Aojun Zhou, Zipeng Qin, Yi Wang, Jifeng Dai, Yu Qiao, Limin Wang, and Hongsheng Li. Journeydb: A benchmark for generative image understanding. In NeurIPS, 2023.

- [43] Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li, Carlos Guestrin, Percy Liang, and Tatsunori B. Hashimoto. Stanford alpaca: An instruction-following llama model. GitHub, 2023.
- [44] Ling Yang, Zhaochen Yu, Bin Cui, and Mengdi Wang. Reasonflux: Hierarchical llm reasoning via scaling thought templates. arXiv preprint arXiv:2502.06772, 2025.

- [45] Yixin Ye, Zhen Huang, Yang Xiao, Ethan Chern, Shijie Xia, and Pengfei Liu. Limo: Less is more for reasoning, 2025.
- [46] Niklas Muennighoff, Zitong Yang, Weijia Shi, Xiang Lisa Li, Li Fei-Fei, Hannaneh Hajishirzi, Luke Zettlemoyer, Percy Liang, Emmanuel Candès, and Tatsunori Hashimoto. s1: Simple test-time scaling, 2025.
- [47] OpenThoughts Team. Open Thoughts. https://open-thoughts.ai, January 2025.
- [48] Zihan Liu, Yang Chen, Mohammad Shoeybi, Bryan Catanzaro, and Wei Ping. Acemath: Advancing frontier math reasoning with post-training and reward modeling. arXiv preprint, 2024.

- [49] Yingzhe Peng, Gongrui Zhang, Miaosen Zhang, Zhiyuan You, Jie Liu, Qipeng Zhu, Kai Yang, Xingzhong Xu, Xin Geng, and Xu Yang. Lmm-r1: Empowering 3b lmms with strong reasoning abilities through two-stage rule-based rl, 2025.
- [50] Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.

- [51] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In ICML, pages 8748–8763, 2021.

- [52] Jiazheng Xu, Xiao Liu, Yuchen Wu, Yuxuan Tong, Qinkai Li, Ming Ding, Jie Tang, and Yuxiao Dong. Imagereward: Learning and evaluating human preferences for text-to-image generation. Advances in Neural Information Processing Systems, 36, 2024.

- [53] Dhruba Ghosh, Hannaneh Hajishirzi, and Ludwig Schmidt. Geneval: An object-focused framework for evaluating text-to-image alignment. In NeurIPS, 2023.

- [54] Yuwei Niu, Munan Ning, Mengren Zheng, Bin Lin, Peng Jin, Jiaqi Liao, Kunpeng Ning, Bin Zhu, and Li Yuan. Wise: A world knowledge-informed semantic evaluation for text-to-image generation. arXiv preprint arXiv:2503.07265, 2025.

- [55] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In CVPR, pages 10684–10695, 2022.

- [56] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with CLIP latents. CoRR, abs/2204.06125, 2022.

- [57] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023.

- [58] Peize Sun, Yi Jiang, Shoufa Chen, Shilong Zhang, Bingyue Peng, Ping Luo, and Zehuan Yuan. Autoregressive model beats diffusion: Llama for scalable image generation. arXiv preprint arXiv:2406.06525, 2024.

- [59] Xianwei Zhuang, Yuxin Xie, Yufan Deng, Liming Liang, Jinghan Ru, Yuguo Yin, and Yuexian Zou. Vargpt: Unified understanding and generation in a visual autoregressive multimodal large language model. arXiv preprint arXiv:2501.12327, 2025.

- [60] Gemini Team, Rohan Anil, Sebastian Borgeaud, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, Katie Millican, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023.

- [61] OpenAI. Openai o1 system card. preprint, 2024.

- [62] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.

- [63] Chunyuan Li, Zhe Gan, Zhengyuan Yang, Jianwei Yang, Linjie Li, Lijuan Wang, Jianfeng Gao, et al. Multimodal foundation models: From specialists to general-purpose assistants. Foundations and Trends® in Computer Graphics and Vision, 16(1-2):1–214, 2024.

- [64] Shukang Yin, Chaoyou Fu, Sirui Zhao, Ke Li, Xing Sun, Tong Xu, and Enhong Chen. A survey on multimodal large language models. arXiv preprint arXiv:2306.13549, 2023.

- [65] Zechen Bai, Pichao Wang, Tianjun Xiao, Tong He, Zongbo Han, Zheng Zhang, and Mike Zheng Shou. Hallucination of multimodal large language models: A survey. arXiv preprint arXiv:2404.18930, 2024.

- [66] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. NeurIPS, 36, 2024.

- [67] Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. Minigpt-4: Enhancing vision-language understanding with advanced large language models. CoRR, abs/2304.10592, 2023.

- [68] Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, et al. Qwen technical report. arXiv preprint arXiv:2309.16609, 2023.

- [69] Brandon McKinzie, Zhe Gan, Jean-Philippe Fauconnier, Sam Dodge, Bowen Zhang, Philipp Dufter, Dhruti Shah, Xianzhi Du, Futang Peng, Floris Weers, et al. Mm1: Methods, analysis & insights from multimodal llm pre-training. arXiv preprint arXiv:2403.09611, 2024.

- [70] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, et al. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024.

- [71] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv:2204.06125, 1(2):3, 2022.

- [72] Junsong Chen, Jincheng Yu, Chongjian Ge, Lewei Yao, Enze Xie, Zhongdao Wang, James T. Kwok, Ping Luo, Huchuan Lu, and Zhenguo Li. Pixart-α: Fast training of diffusion transformer for photorealistic text-to-image synthesis. In ICLR. OpenReview.net, 2024.

- [73] Alex Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela Mishkin, Bob McGrew, Ilya Sutskever, and Mark Chen. Glide: Towards photorealistic image generation and editing with text-guided diffusion models. arXiv preprint arXiv:2112.10741, 2021.

- [74] Zeyue Xue, Guanglu Song, Qiushan Guo, Boxiao Liu, Zhuofan Zong, Yu Liu, and Ping Luo. Raphael: Text-toimage generation via large mixture of diffusion paths. NeurIPS, 36, 2024.

- [75] Ling Yang, Zhaochen Yu, Chenlin Meng, Minkai Xu, Stefano Ermon, and Bin Cui. Mastering text-to-image diffusion: Recaptioning, planning, and generating with multimodal llms. In Forty-first International Conference on Machine Learning, 2024.

- [76] Xinchen Zhang, Ling Yang, Guohao Li, Yaqi Cai, Jiake Xie, Yong Tang, Yujiu Yang, Mengdi Wang, and Bin Cui. Itercomp: Iterative composition-aware feedback learning from model gallery for text-to-image generation. In ICLR, 2025.

- [77] Minshuo Chen, Song Mei, Jianqing Fan, and Mengdi Wang. An overview of diffusion models: Applications, guided generation, statistical rates and optimization. arXiv preprint arXiv:2404.07771, 2024.

- [78] Ye Tian, Ling Yang, Xinchen Zhang, Yunhai Tong, Mengdi Wang, and Bin Cui. Diffusion-sharpening: Fine-tuning diffusion models with denoising trajectory sharpening. arXiv preprint arXiv:2502.12146, 2025.

- [79] Hui Yuan, Kaixuan Huang, Chengzhuo Ni, Minshuo Chen, and Mengdi Wang. Reward-directed conditional diffusion: Provable distribution estimation and reward improvement. Advances in Neural Information Processing Systems, 36:60599–60635, 2023.

- [80] Yingqing Guo, Hui Yuan, Yukang Yang, Minshuo Chen, and Mengdi Wang. Gradient guidance for diffusion models: An optimization perspective. arXiv preprint arXiv:2404.14743, 2024.

- [81] Minshuo Chen, Kaixuan Huang, Tuo Zhao, and Mengdi Wang. Score approximation, estimation and distribution recovery of diffusion models on low-dimensional data. In International Conference on Machine Learning, pages 4672–4712. PMLR, 2023.

- [82] Fu-Yun Wang, Ling Yang, Zhaoyang Huang, Mengdi Wang, and Hongsheng Li. Rectified diffusion: Straightness is not your need in rectified flow. arXiv preprint arXiv:2410.07303, 2024.

- [83] Ling Yang, Jingwei Liu, Shenda Hong, Zhilong Zhang, Zhilin Huang, Zheming Cai, Wentao Zhang, and Bin Cui. Improving diffusion-based image synthesis with context prediction. Advances in Neural Information Processing Systems, 36:37636–37656, 2023.

- [84] Ling Yang, Haotian Qian, Zhilong Zhang, Jingwei Liu, and Bin Cui. Structure-guided adversarial training of diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7256–7266, 2024.

- [85] Ye Tian, Ling Yang, Haotian Yang, Yuan Gao, Yufan Deng, Xintao Wang, Zhaochen Yu, Xin Tao, Pengfei Wan, Di ZHANG, et al. Videotetris: Towards compositional text-to-video generation. Advances in Neural Information Processing Systems, 37:29489–29513, 2024.

- [86] Jacob Austin, Daniel D Johnson, Jonathan Ho, Daniel Tarlow, and Rianne Van Den Berg. Structured denoising diffusion models in discrete state-spaces. NeurIPS, pages 17981–17993, 2021.

- [87] Shuyang Gu, Dong Chen, Jianmin Bao, Fang Wen, Bo Zhang, Dongdong Chen, Lu Yuan, and Baining Guo. Vector quantized diffusion model for text-to-image synthesis. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10696–10706, 2022.

- [88] Emiel Hoogeboom, Alexey A. Gritsenko, Jasmijn Bastings, Ben Poole, Rianne van den Berg, and Tim Salimans. Autoregressive diffusion models. In ICLR. OpenReview.net, 2022.

- [89] Kevin P. Murphy. Probabilistic Machine Learning: Advanced Topics. MIT Press, 2023.

- [90] Patrick Esser, Robin Rombach, and Bjorn Ommer. Taming transformers for high-resolution image synthesis. In CVPR, pages 12873–12883, 2021.

- [91] Yuchao Gu, Xintao Wang, Yixiao Ge, Ying Shan, and Mike Zheng Shou. Rethinking the objectives of vectorquantized tokenizers for image synthesis. In CVPR, pages 7631–7640, 2024.

- [92] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. NeurIPS, 30, 2017.

- [93] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of machine learning research, 21(140):1–67, 2020.

- [94] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023.

- [95] Niki Parmar, Ashish Vaswani, Jakob Uszkoreit, Lukasz Kaiser, Noam Shazeer, Alexander Ku, and Dustin Tran. Image transformer. In ICML, pages 4055–4064, 2018.

- [96] Patrick Esser, Robin Rombach, and Bjorn Ommer. Taming transformers for high-resolution image synthesis. In CVPR, pages 12873–12883, 2021.

- [97] Suman Ravuri and Oriol Vinyals. Classification accuracy score for conditional generative models. NeurIPS, 32, 2019.

- [98] Mark Chen, Alec Radford, Rewon Child, Jeffrey Wu, Heewoo Jun, David Luan, and Ilya Sutskever. Generative pretraining from pixels. In ICML, pages 1691–1703, 2020.

- [99] Dan Kondratyuk, Lijun Yu, Xiuye Gu, José Lezama, Jonathan Huang, Rachel Hornung, Hartwig Adam, Hassan Akbari, Yair Alon, Vighnesh Birodkar, et al. Videopoet: A large language model for zero-shot video generation. arXiv preprint arXiv:2312.14125, 2023.

- [100] Keyu Tian, Yi Jiang, Zehuan Yuan, Bingyue Peng, and Liwei Wang. Visual autoregressive modeling: Scalable image generation via next-scale prediction. Advances in neural information processing systems, 37:84839–84865, 2024.

- [101] Shengqiong Wu, Hao Fei, Leigang Qu, Wei Ji, and Tat-Seng Chua. Next-gpt: Any-to-any multimodal llm. arXiv preprint arXiv:2309.05519, 2023.

- [102] Zineng Tang, Ziyi Yang, Chenguang Zhu, Michael Zeng, and Mohit Bansal. Any-to-any generation via composable diffusion. NeurIPS, 36, 2024.

- [103] Hanrong Ye, De-An Huang, Yao Lu, Zhiding Yu, Wei Ping, Andrew Tao, Jan Kautz, Song Han, Dan Xu, Pavlo Molchanov, et al. X-vila: Cross-modality alignment for large language model. arXiv preprint arXiv:2405.19335, 2024.

- [104] Emanuele Aiello, LILI YU, Yixin Nie, Armen Aghajanyan, and Barlas Oguz. Jointly training large autoregressive multimodal models. In ICLR, 2024.

- [105] Shuyang Gu, Dong Chen, Jianmin Bao, Fang Wen, Bo Zhang, Dongdong Chen, Lu Yuan, and Baining Guo. Vector quantized diffusion model for text-to-image synthesis. In CVPR, pages 10696–10706, 2022.

- [106] Huiwen Chang, Han Zhang, Lu Jiang, Ce Liu, and William T Freeman. Maskgit: Masked generative image transformer. In CVPR, pages 11315–11325, 2022.

- [107] Huiwen Chang, Han Zhang, Jarred Barber, AJ Maschinot, Jose Lezama, Lu Jiang, Ming-Hsuan Yang, Kevin Murphy, William T Freeman, Michael Rubinstein, et al. Muse: Text-to-image generation via masked generative transformers. arXiv preprint arXiv:2301.00704, 2023.

- [108] Emiel Hoogeboom, Didrik Nielsen, Priyank Jaini, Patrick Forré, and Max Welling. Argmax flows and multinomial diffusion: Learning categorical distributions. Advances in neural information processing systems, 34:12454–12465, 2021.

- [109] Jacob Austin, Daniel D Johnson, Jonathan Ho, Daniel Tarlow, and Rianne Van Den Berg. Structured denoising diffusion models in discrete state-spaces. Advances in neural information processing systems, 34:17981–17993, 2021.

- [110] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.

- [111] John Schulman, Philipp Moritz, Sergey Levine, Michael Jordan, and Pieter Abbeel. High-dimensional continuous control using generalized advantage estimation. arXiv preprint arXiv:1506.02438, 2015.

## Appendix

### A Preliminaries of Discrete Diffusion, PPO and GRPO

#### A.1 Discrete Diffusion and Mask Token Prediction

Discrete denoising diffusion models have emerged as a powerful paradigm for modeling discrete data representations, particularly in visual and textual domains. Recent works such as VQ-Diffusion [105], MaskGIT [106], and Muse [107] demonstrate their effectiveness in generating high-quality discrete tokens. Building on these advancements, Show-o [20] unifies discrete diffusion and mask token prediction into a single framework, enabling joint optimization of noise corruption and semantic recovery. Below, we formalize the key components of this discrete diffusion process.

###### A.1.1 Forward and Reverse Diffusion Processes

The forward diffusion process progressively corrupts the initial data x0 (e.g., a sequence of visual/textual tokens) into noisy latent variables x1,...,xT via a fixed Markov chain q(xt|xt−1). At each step, this process stochastically perturbs tokens—such as replacing them with uniform noise or introducing a special [MASK] token. The reverse process, parameterized by a learned model pθ, reconstructs x0 from xT by sequentially sampling q(xt−1|xt,x0).

To formalize this, consider a single token xi0 at position i in x0, where xi0 ∈ {1,2,...,K} corresponds to an entry in a codebook. For simplicity, we omit the index i in subsequent derivations. The transition probabilities

between consecutive tokens are defined by a matrix Qt ∈ RK×K, with [Qt]mn = q(xt = m|xt−1 = n). The forward Markov process for the entire token sequence is then expressed as:

q(xt|xt−1) = v⊤(xt)Qtv(xt−1), (7)

where v(x) is a one-hot vector of length K (with 1 at position x). The categorical distribution over xt is derived from Qtv(xt−1).

A critical property of the Markov chain allows us to marginalize intermediate steps and compute the direct transition from x0 to xt:

q(xt|x0) = v⊤(xt)Qtv(x0), with Qt = QTQT−1 ···Q1. (8)

Furthermore, the posterior distribution conditioned on x0 is analytically tractable:

v⊤(xt)Qtv(xt−1) v⊤(xt−1)Qt−1v(x0) v⊤(xt)Qtv(x0)

q(xt|xt−1,x0)q(xt−1|x0) q(xt|x0)

. (9)

q(xt−1|xt,x0) =

=

This property is essential for deriving the variational lower bound of the diffusion process.

###### A.1.2 Transition Matrix Design and Masking Strategy

The design of the transition matrix Qt is pivotal to the success of discrete diffusion models. Early works [108, 109] propose injecting uniform noise into the categorical distribution, leading to:





αt + βt βt ··· βt βt αt + βt ··· βt

, (10)

Qt =

 

 

... . βt βt ··· αt + βt

. .

where αt ∈ [0,1] controls the retention probability, and βt = (1 − αt)/K ensures uniform diffusion across all K categories. However, this approach often introduces abrupt semantic changes due to aggressive replacement of tokens with unrelated categories.

To address this limitation, Show-o adopts a mask-and-replace strategy inspired by masked language modeling.

- A special [MASK] token is introduced, expanding the token space to K + 1 states. The transition matrix is redefined as:





αt + βt βt ··· βt 0 βt αt + βt ··· βt 0

... . . βt βt ··· αt + βt 0 γt γt ··· γt 1

, (11)

Qt =

. .

 

 

where: - Ordinary tokens have a probability αt to remain unchanged, βt to be uniformly diffused, and γt to be replaced by [MASK]. - The [MASK] token retains its state with probability 1.

This design explicitly signals corrupted positions during the forward process, enabling the reverse network to focus on reconstructing masked regions. The parameters satisfy αt + Kβt + γt = 1, ensuring proper normalization.

###### A.1.3 Variational Objective and Loss Simplification

Following the D3PM framework [109], the variational lower bound for the discrete diffusion process is derived as:

T

0,xt)[log pθ(x0|xt)] + C. (12)

Eq(x

Eq(x

0)[log pθ(x0)] ≥ −LELBO(x0,θ) ≥

t=1

To simplify training, Show-o leverages the mask token prediction objective. Specifically, the model learns a neural network pθ to reconstruct masked tokens in x0 from the noised xt, following the methodology of MaskGIT [106]. This approach avoids explicit modeling of the full posterior and instead focuses on recovering only the corrupted regions, significantly reducing computational complexity.

#### A.2 Proximal Policy Optimization and Group Relative Policy Optimization

Proximal Policy Optimization (PPO) [110] is a widely adopted reinforcement learning algorithm that balances policy updates with stability through a clipped surrogate objective. The core idea of PPO lies in constraining policy changes to a proximal region around the previous policy, thereby preventing large, destabilizing updates. This is achieved by introducing a clipping mechanism that bounds the importance sampling ratio during optimization. Specifically, PPO maximizes the following objective:

πθ(ot | q,o<t) πθ

πθ(ot | q,o<t) πθ

Aˆt, clip

,1 − ε,1 + ε A ˆt , (13)

JPPO(θ) = E(q,a)∼D,o

≤t∼πθold(·|q) min

(ot | q,o<t)

(ot | q,o<t)

old

old

where (q,a) denotes a question-answer pair sampled from the dataset D, ε is the clipping threshold, and Aˆt is the estimated advantage at time step t. The advantage Aˆt is typically computed using Generalized Advantage Estimation (GAE) [111], which combines multiple temporal differences with discounting and mixing parameters (γ,λ):

AˆGAEt (γ,λ) =

∞

(γλ)lδt+l, with δl = Rl + γV (sl+1) − V (sl). (14)

l=0

This formulation ensures robustness against high-variance estimates while maintaining compatibility with off-policy data.

In contrast, Group Relative Policy Optimization (GRPO) [27] introduces two key innovations to address limitations in PPO’s value function dependency and global advantage estimation. First, GRPO eliminates the explicit modeling of the value function V (s) and instead computes advantages in a group-relative manner. For each (q,a) pair, the behavior policy πθ

generates G responses {oi}Gi=1, and the advantage of each response is normalized relative to its group:

old

ri − mean({Ri}Gi=1) std({Ri}Gi=1)

Aˆi,t =

, (15)

where ri represents the raw reward for response oi. This design enables GRPO to focus on relative performance within a local context, reducing sensitivity to absolute reward scales.

Second, GRPO extends PPO’s clipped objective by incorporating an explicit KL divergence penalty term between the current policy πθ and a reference policy πref. The final objective is defined as:

JGRPO(θ) = E(q,a)∼D,{o

i}Gi=1∼πθold(·|q)

|oi|

G

1 G

1 |oi|

min ri,t(θ)Aˆi,t, clip ri,t(θ),1 − ε,1 + ε A ˆi,t − βDKL(πθ||πref) ,

t=1

i=1

(16)

where ri,t(θ) = ππθ(oi,t|q,oi,<t)

θold(oi,t|q,oi,<t) is the importance sampling ratio at time t. Notably, GRPO computes the loss

- at the sample level: for each sequence, the per-token loss is averaged first, followed by averaging across all sequences in the group. This hierarchical aggregation enhances stability in multi-response settings.

### B Details of UniGRPO

In this section, we detail our UniGRPO training process and illustrate the differences compared to LLaDA and d1 (diff-GRPO) [28]. Notably, LLaDA does not incorporate a reinforcement learning procedure, yet it includes an algorithm for computing log probabilities. Diff-GRPO introduced the first GRPO-style RL algorithm for diffusion language models and proposed an alternative method to compute log probabilities for a given question-answer pair. We first outline the methodologies of these two prior works:

- (1) LLaDA While LLaDA does not employ an RL process, it offers a method to estimate the log probability of a given pair (q,a) using Monte Carlo simulation. For a given answer, N mask ratios are randomly sampled from the interval (0,1) (where N = 128 in the original work). Subsequently, a batch of N inputs is constructed. Each input instance consists of the original question tokens concatenated with the answer, where a portion of the answer tokens (corresponding to the sampled mask ratio) is replaced by [MASK] tokens. A model forward pass is performed on this batch, and the logits corresponding to the masked regions are collected and averaged to obtain the log probability log p(q,a). The principal drawback of this method is its significant computational overhead, rendering it impractical for on-policy RL algorithms.
- (2) d1 (diff-GRPO) The d1 framework [28] introduced the first GRPO-style RL algorithm for diffusion language models. To overcome the efficiency issues of LLaDA, d1 employs a novel masking strategy. In each iteration, only a single forward pass is performed, eschewing Monte Carlo simulation. The input is constructed by applying a random mask to the question tokens and completely masking all answer tokens. The stochasticity is intended to be achieved by selecting different tokens for masking across different iterations, even with the same mask ratio. While this approach in d1 enables GRPO training, we identify potential limitations:

- • Question Masking: The random masking of question tokens is of unclear practical significance. In typical question-answering scenarios, the question is always fully observed during both training and inference. We argue that such random masking of questions serves primarily to introduce stochasticity without direct relevance to the task’s practical application.
- • Answer Masking Strategy: By consistently masking the entire answer, the model is effectively trained only on the initial denoising step (i.e., predicting the full sequence from a fully masked state). We contend that this approach provides insufficient learning depth, potentially causing the model to behave more like a single-step predictive AR model rather than leveraging the multi-step denoising capabilities inherent to diffusion models. This underutilizes the diffusion model’s unique strengths.

- As introduced in section 2.3.1, our UniGRPO addresses the masking issues with the following key modifications:

- 1. Unmasked Questions: In UniGRPO, all question tokens remain unmasked during training. This aligns the training process with the inference conditions and practical use-cases where the question is always fully provided.

- 2. Iteratively Varied Answer Masking: We apply a random mask ratio to the answer tokens. Crucially, this

mask ratio, denoted ρa, varies uniformly with the training iteration µ (e.g., ρa ∼ U(0,1) sampled anew or cycled through discrete steps each iteration). This strategy maintains stochasticity while exposing the model to diverse denoising stages—from nearly fully masked to almost fully denoised answers. This enables UniGRPO to learn from multi-step denoising, aligning with standard diffusion model training and leveraging their full generative potential across multiple steps.

- At each iteration, we sample a mask ratio rµ uniformly from a predefined range, apply this ratio to the answer tokens, and perform a single forward pass using the unmasked question and masked answer. We then compute token-level log-likelihoods in the masked regions, apply the GRPO objective, and update the policy accordingly.

### C Qualitative Comparisons (with Reasoning CoT)

- Qualitative Comparison of World Knowledge-Aware Text-to-Image Generation (1) Prompt: The largest terrestrial carnivore from the Arctic.

Other models:

Show-o Emu3 Janus Pro 7B

[Figure 40]

[Figure 41]

[Figure 42]

MMaDA: <think>

largest terrestrial carnivore from the Arctic is the Polar Bear. </think> A majestic polar bear stands on a snowy landscape, its fur glowing against the icy backdrop.

[Figure 43]

- Qualitative Comparison of World Knowledge-Aware Text-to-Image Generation (2) Prompt: A famous landmark building in Europe.

Other models:

Show-o Emu3 Janus Pro 7B

[Figure 44]

[Figure 45]

[Figure 46]

MMaDA: <think>

A famous landmark building in Europe is likely the Eiffel Tower. </think> The Eiffel Tower is hollow, with a cloudy and muted sky.

[Figure 47]

- Qualitative Comparison of World Knowledge-Aware Text-to-Image Generation (3) Prompt: A traditional strong spirit from Russia.

Other models:

Show-o Emu3 Janus Pro 7B

[Figure 48]

[Figure 49]

[Figure 50]

MMaDA: <think>

The user is describing a traditional strong spirit from Russia. A well-known spirit that fits this description is vodka.

</think> An silver bottle of vodka sits next to a decanter and a halved lime.

[Figure 51]

Qualitative Comparison of Textual Reasoning (1) Question: How many plums will balance one pear, given that 3 apples and one pear weigh as much as 10 plums, and one apple and 6 plums balance one pear? Assume that fruits of the same kind have the same weight. Show-o:

< Show-o does not support this task >

Emu3:

< Emu3 does not support this task >

Janus Pro 7B:

< Janus Pro 7B does not support this task >

MMaDA: <think>

Okay, let’s see. I need to figure out how many plums are needed to balance the weight of a single pear. The problem gives me some information. Let me try to parse it step by step. First, the problem says: 3 apples together with a pear have the same weight as 10 plums. Then, the combination of one apple and 6 plums weighs the same as a pear. I need to find out how many plums are needed to equal the weight of a single pear. Since all fruits are equal in weight, probably the same. Let me start by writing this down. Let’s assign variables. Let’s let A represent the weight of an apple, R as the weight of a pear, and M as the weight of a plum. So, based on the first statement given: 3A + R = 10M Right, because 3 apples plus a pear equals 10 plums. Second, saying one apple and 6 plums weigh the same as a pear. So, A + 6M = R That’s because all apples and plums are equal. So, A = R - 6M So, now we have two equations:

- 1) 3A + R = 10M
- 2) A + 6M = R Our goal is to find out how many plums are needed to balance one pear. That is, we need to find the value of n in R = nM. From equation (2): A + 6M = R =⇒ A = R − 6M.

Now substitute this value of A into equation (1): 3A + R = 10M 3(R - 6M) + R = 10M 3R - 18M + R = 10M 4R - 18M = 10M 4R = 28M R = 7M So one pear weighs as much as 7 plums. Let’s check with equation (2): A + 6M = R A = R - 6M But if R = 7M, then A = 7M − 6M = M. So an apple weighs as much as 1 plum, which works when we check the first equation: 3A + R = 10M 3(M) + 7M = 10M 3M + 7M = 10M So both equations are satisfied. Therefore, the answer is 7 plums.

###### </think>

To determine how many plums are to balance a single pear, we set up the equations based on the given conditions:

- 1. Let A represent the weight of an apple, R represent the weight of a pear, and M the weight of a plum.
- 2. The equations are:

- - Equation 1: 3A + R = 10M
- - Equation 2: A + 6M = R (1 apple + 6 plums equals a pear)

- **Step 1: Solve for A in terms of R and M** From Equation 2: A = R − 6M
- **Step 2: Substitute A into Equation 1**

- 3(R − 6M) + R = 10M

- 3R − 18M + R = 10M
- 4R − 18M = 10M

- 4R = 28M R = 7M

**Final Answer:** plums are needed to balance the weight of a single pear.

|7|
|---|

Question: How many positive divisors does the integer n possess, where n = 73 × 112 × 134?

Show-o:

< Show-o does not support this task >

Emu3:

< Emu3 does not support this task >

Janus Pro 7B:

< Janus Pro 7B does not support this task >

MMaDA: <think>

Okay, so I need to figure out how many positive divisors the integer n has, where n is given as 7 cubed times 11 squared times 13 to the fourth power. Hmm, let me think. I remember that the number of divisors of a number is related to its prime factorization exponents. Let me try to recall the general formula here.

First, right that if a number is expressed in a product of primes, like a number in the form n = pa1 × pb2 × . . . × pck, then the number of positive divisors is (a + 1)(b + 1)...(c + 1). Right? Because for each exponent in the prime

factorization, you add 1, then multiply them all together for the total number of divisors. Let me verify that. Take an example: if n is 21 × 31, then the positive divisors are 1, 2, 3, 6. Using the formula, (1+1)*(1+1) = 2*2 = 4, which checks out. Another example: like 22 × 31. Then the divisors are 1, 2, 4, 3, 6, 12. That makes 6 divisors. The formula gives (2+1)*(1+1) = 3*2 = 6, so that’s correct. Another example: 22 × 52. Then divisors count is (2+1)*(2+1) = 3*3=9. So the formula seems correct. So, applying this formula to n in the given problem: n is 73 × 112 × 134. Here, the exponents are 3, 2, and 4. So, applying the number of divisors formula should be (3 + 1)(2 + 1)(4 + 1). Let’s compute that step by step. First, calculate each exponent plus one: 3 + 1 is 4. 2 + 1 is 3. 4 + 1 is 5. Then, multiplying those results together: 4 * 3 * 5. Let me check that. 4 times 3 is 12, then 12 times 5 is 60. So the number of positive divisors is 60. Wait, but let me make sure I didn’t make a mistake in the multiplication. Let me verify the steps again. 4*3 is 12, 12*5 is 60. Yes, that seems correct. But just to be thorough, let me write the original number n as 73 × 112 × 134. The prime factors are 7, 11, 13. Each raised to the exponents 3, 2, and 4, respectively. Since they are all prime, the formula for divisors uses the exponents as factors plus 1. Then multiplying the results: 4*3*5=60. Is there any possible way this could be wrong? Well, maybe if the exponents were different, but the original expression is already in its exponential form. Let me check if I misread. For example, if one of the exponents was not what is stated, but the problem writes 73 times 112 times 134, so the exponents are 3, 2, 4, that’s correct. Alternatively, could there be a mistake in the multiplication? Let me check 4*3*5 again. 4*3 is 12, 12*5 is 60. Correct. So 60 is the correct answer. Wait, just to be thorough, let’s list the combinations: any divisor is 7a × 11b × 13c, where 0 ≤ a ≤ 3, 0 ≤ b ≤ 2, and 0 ≤ c ≤ 4. The number of choices for a is 4, for b is 3, for c is 5, so 4*3*5 = 60 possible positive divisors. Perfect. I don’t see any mistakes here. The answer is 60.

###### </think>

**Solution:** The integer n is given as n = 73 × 112 × 134. To determine the number of positive divisors of n, we use the formula

for the number of divisors: if a number is in the form pa1 × pb2 × · · · × pck, then it has (a + 1)(b + 1) . . . (c + 1) divisors.

**Step-by-Step Solution:**

- 1. **Identify the Exponent:**

- - 73: Exponent = 3
- - 112: Exponent = 2
- - 134: Exponent = 4

- 2. **Apply the Divisor Formula:**

- - For each prime exponent, (a + 1), and then multiply:
- - For exponent 3: 3 + 1 = 4
- - For exponent 2: 2 + 1 = 3
- - For exponent 4: 4 + 1 = 5
- - Multiply the results: 4 × 3 × 5 = 60

|60|
|---|

positive divisors.

**Answer:** The integer n has

Question: A "Hope Number" is a natural number with an odd number of positive divisors. Find the largest "Hope Number" within 1000.

Show-o:

< Show-o does not support this task >

Emu3:

< Emu3 does not support this task >

Janus Pro 7B:

< Janus Pro 7B does not support this task >

MMaDA: <think>

Okay, so I need to find the largest "Hope Number" which is a natural number with an odd number of positive divisors. Let me start by recalling what a Hope Number is. The question says a number has an odd number of divisors. Hmm, that means a number has an odd number of different divisors. First, I remember that numbers have divisors related to their prime factorization. If a number has the prime

factorization n = pa11pa22 . . . pakk, the number of divisors is (a1 + 1)(a2 + 1) · · · (ak + 1). This product is odd if and only if every ai + 1 is odd, i.e., if every exponent ai is even. That means the number itself is a perfect square (since each exponent is even).

To check: for example, take 9. Its prime factorization is 32. Its divisors are 1, 3, 9. That’s three divisors, which is odd. Another example, 16. Prime factorization is 24. Its divisors are 1, 2, 4, 8, 16. That’s five divisors, which is odd. How about 15? Its prime factorization is 3 × 5, which is not a perfect square. Its divisors are 1, 3, 5, 15—that’s four divisors, which is even. Therefore, only perfect squares have an odd number of divisors. Therefore, all Hope Numbers less than 1000 are perfect squares. Therefore, the problem reduces to finding the largest perfect square less than or equal to 1000. Let’s compute that. The square root of 1000 is approximately 31.62. So we need to check the largest integer n such that n2 ≤ 1000. Let’s check 31. 312 = 961. 322 = 1024, which is greater than 1000. So 312 = 961 is the largest perfect square within 1000. Therefore, 961 is the largest Hope Number within 1000. But wait, let me confirm that 961 is indeed a Hope Number. Let’s count its divisors. Its prime factorization is 312, and 31 is prime. The exponents are all even, so the number of divisors is 2 + 1 = 3: 1, 31, 961, which is odd. That checks out. And to be thorough, is there a perfect square between 961 and 1000? 322 = 1024 > 1000, so there are none. Therefore, the largest Hope Number within 1000 is 961.

###### </think>

The largest "Hope Number" within 1000 is **961**.

**Step-by-Step Explanation:**

- 1. **Understanding Hope Property:** A Hope Number has an odd number of positive divisors. A number has an odd number of divisors **if and only if it is a perfect square** (because in the divisor formula, all exponents must be even so all ai + 1 are odd, and their product is odd).
- 2. **Largest Hope Number:** Compute perfect squares ≤ 1000. 312 = 961 and 322 = 1024. Since 322 > 1000, the largest one is 312 = 961.
- 3. **Verification:** Any non-square number below 1000 has an even number of divisors; only perfect squares have an odd number of divisors.

**Answer:** 961

[Figure 52]

Question: In the given figure, a square ABCD is inscribed in a circle with center O. Point P is located on side CD. What is the value of angle APB?

Show-o: The angle APB is 90 degrees.

Emu3: The question seems to be a mix of letters and numbers, not a coherent question or statement. If you meant "is" or "are", the correct statement would be "The measure of angle APB is" or "The measure of angle APB is". If you meant "is" or "are", the correct statement would be "The measure of angle APB is" or "The measure of angle APB is".

###### Janus Pro 7B:

To find the measure of angle APB, we can use the properties of the square and the circle.

- 1. **Square ABCD**: The square is circumscribed around the circle, meaning all its sides are tangent to the circle. This implies that the diagonal of the square is equal to the diameter of the circle.
- 2. **Circle O**: The center of the circle is also the center of the square.
- 3. **Diagonals of a Square**: The diagonals of a square bisect each other and are equal in length.
- 4. **Angle APB**: Since ABCD is a square, angle APB is a right angle (90 degrees). Therefore, the measure of angle APB is 90 degrees.

###### MMaDA: <think>

To determine the measure of angle APB, we start by noting that square ABCD is circumscribed within circle O. This means that the circle is the circumcircle of the square, and the center of the circle is the same as the center of the square. Let’s denote the center of the circle as O. Since ABCD is a square, the diagonals AC and BD are equal in length and bisect each other at right angles. This means that angle AOB is 90 degrees because the diagonals of a square are equal and bisect each other at right angles. Now, consider triangle APB. Since P lies on side CD, we can use the property of the circle that the angle subtended by a chord at the center is twice the angle subtended at any point on the circumference on the same side. In this case, AP and BP are chords of the circle meeting at point P on CD (which is opposite AB). Points A and B are at adjacent vertices, and P is on CD. The key point is that when P moves along CD, angle APB is fixed, as AB is a fixed chord and P moves along the arc opposite AB. For a square inscribed in a circle, the arc AB subtends a right angle at any point on the circle opposite AB (i.e., on arc CD). Therefore, for any point P on CD, angle APB is always 45°. This follows because ∠APB is half the degree of the arc AB, which is 90°, so:

- 1

- 2 × 90◦ = 45◦

∠APB =

|45◦|
|---|

Thus, the measure of angle APB is

.

###### </think>

|45◦|
|---|

.

The measure of angle APB is

[Figure 53]

###### Question: What number of large things are the same material as the purple cylinder?

Show-o: There are two large things that are the same material as the purple cylinder.

###### Emu3: 2

Janus Pro 7B: There is one large object that shares the same material as the purple cylinder, which is the large green cylinder. Answer: 1.

###### MMaDA: <think>

To determine the number of items in the image, let’s analyze the objects present:

- 1. The large green cylinder is made of a material that appears to be rubber.
- 2. The large yellow cube is made of a material that appears to be metal.
- 3. The small gray cube is made of a material that appears to be rubber.
- 4. The small cyan sphere is made of a material that appears to be metal.
- 5. The small cyan cylinder is made of a material that appears to be metal.
- 6. The small purple cylinder is made of a material that appears to be rubber. The green cylinder is the only large object made of a material that is the same as the small purple cylinder. Therefore, there is 1 large object made of the same material as the purple cylinder. </think> The number of large things that are the same material as the purple cylinder is 1.

Qualitative Comparison of Multimodal Reasoning (4)

[Figure 54]

Question: Divide 24 apples equally among the people in the picture, how many apples can each person get?

Show-o: In the image, there are four people, each with a different number of apples. The person with the most apples has the most apples.

Emu3: 24.

Janus Pro 7B:

To divide 24 apples equally among the four people in the picture, you would divide 24 by 4.

24 / 4 = 6 Each person can get 6 apples.

###### MMaDA: <think>

To determine how many apples each person can get, we need to divide the total number of apples by the number of people. 24 ÷ 4 = 6 So, each person can get 6 apples.

</think> Each person can get 6 apples.

- D Qualitative Examples (without Reasoning CoT) MMaDA can also efficiently handle general and simple tasks without generating CoT.

Qualitative Example of Text-to-Image Generation

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

- Qualitative Example of Multimodal Understanding (1)

[Figure 70]

Prompt: Please describe this image in detail.

MMaDA:

The image captures a serene and picturesque scene of a group of dolphins swimming in the ocean during sunset. The sky is painted with hues of orange, yellow, and pink, transitioning into a deep blue as it meets the horizon. The sun is setting on the horizon, casting a warm glow that reflects off the water’s surface, creating a shimmering effect. The dolphins are positioned in the foreground, with their bodies partially submerged in the water, their heads and upper bodies visible above the surface. The water appears calm, with gentle ripples around the dolphins, adding to the tranquil atmosphere. The overall composition of the image conveys a sense of peace and natural beauty.

- Qualitative Example of Multimodal Understanding (2)

[Figure 71]

Prompt: Please describe this image in detail.

MMaDA:

The image is a vibrant and colorful landscape painting that captures the essence of a sunset over a field of yellow flowers. The sky is a stunning blend of warm, soft hues, transitioning from deep purples and blues at the top to warm oranges and yellows near the horizon. The sun is positioned low on the horizon, casting a golden glow that illuminates the sky and the field below. The field of yellow flowers, likely sunflowers, stretches across the foreground, creating a sea of color that contrasts beautifully with the cooler tones of the sky. The painting is executed with broad, expressive brushstrokes, giving it a dynamic and lively feel. The overall composition is balanced and harmonious, capturing the tranquility and beauty of the natural world.

- Qualitative Example of Multimodal Understanding (3)

[Figure 72]

Prompt: Please describe this image in detail.

MMaDA:

The image depicts a modern dining area with a minimalist aesthetic. The centerpiece is a long dining table with a sleek, dark finish, supported by four sturdy legs. The table is set with a few items, including a black vase, a few books, and a small decorative object, adding a touch of elegance to the space. The table is surrounded by four blue velvet chairs with wooden legs, providing a striking contrast to the dark tones of the table. The room has a clean, open feel with light-colored walls and large windows that allow natural light to flood in, creating a bright and airy atmosphere. The left is a round mirror with a simple frame, enhancing the modern look of the room.

- Qualitative Example of Multimodal Understanding (4)

[Figure 73]

Prompt: Please describe this image in detail.

MMaDA:

The image depicts a charming, narrow street in a European city, likely in, given the architectural style and the presence of a red tram car. The street is lined with tall, narrow buildings painted in soft pastel colors, including shades of pink, green, and beige. The buildings have ornate balconies and shutters, adding to the quaint charm of the scene. The street is paved with cobblestones, and there tram tracks run down the center, indicating that this is a tram lane. A red tram car is parked on the right. The tram is stationary, and the overall atmosphere is serene and picturesque.

