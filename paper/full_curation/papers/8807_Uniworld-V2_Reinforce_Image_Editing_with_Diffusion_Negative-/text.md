# arXiv:2510.16888v3[cs.CV]4Nov2025

[Figure 1]

## UNIWORLD-V2: REINFORCE IMAGE EDITING WITH DIFFUSION NEGATIVE-AWARE FINETUNING AND MLLM IMPLICIT FEEDBACK

UniWorld Team 1Shenzhen Graduate School, Peking University 2Rabbitpre AI Full author list in Contributions

Github HuggingFace

[Figure 2]

ABSTRACT

Instruction-based image editing has achieved remarkable progress; however, models solely trained via supervised fine-tuning often overfit to annotated patterns, hindering their ability to explore and generalize beyond training distributions. To this end, we introduce Edit-R1, a novel post-training framework for instructionbased image editing based on policy optimization. Specifically, we utilize Diffusion Negative-aware Finetuning (DiffusionNFT), a likelihood-free policy optimization method consistent with the flow matching forward process, thereby enabling the use of higher-order samplers and more efficient training. Another key challenge here is the absence of a universal reward model, resulting from the diverse nature of editing instructions and tasks. To bridge this gap, we employ a Multimodal Large Language Model (MLLM) as a unified, training-free reward model, leveraging its output logits to provide fine-grained feedback. Furthermore, we carefully design a low-variance group filtering mechanism to reduce MLLM scoring noise and stabilize optimization. UniWorld-V2, trained with this framework, achieves state-of-the-art results on the ImgEdit and GEdit-Bench benchmarks, scoring 4.49 and 7.83, respectively. Crucially, our framework is modelagnostic, delivering substantial performance gains when applied to diverse base models like Qwen-Image-Edit and FLUX-Kontext, demonstrating its wide applicability. Code and models are publicly available to support further research.

[Figure 3]

[Figure 4]

7.83

4.49 4.48

7.76

7.56 7.54 7.53

4.35

4.27

4.2

6.74

4.02

4.0

6.56

6.0

3.71

GPT-Image-1 [High]UniWorld-FLUX.1-

FLUX.1-Kontext

GPT-Image-1 [High]

FLUX.1-Kontext

Qwen-Image-Edit

FLUX.1-Kontext

Qwen-Image-Edit

Qwen-Image-Edit

FLUX.1-Kontext

Qwen-Image-Edit

UniWorld-FLUX.1-

UniWorld-V2UniWorld-Qwen-Image-Edit [2509]

UniWorld-V2UniWorld-Qwen-Image-Edit [2509]

Kontext [dev]

Kontext [dev]

[dev]

[dev]

[2509]

[2509]

[pro]

[pro]

Figure 1: On ImgEdit (Ye et al., 2025b) and GEdit-Bench (Liu et al., 2025b) leaderboards, our method achieves state-of-the-art performance.

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

Adjust Background

[Figure 10]

[Figure 11]

[Figure 12]

Poster Edit

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

Remove

[Figure 17]

Remove

[Figure 18]

[Figure 19]

[Figure 20]

Hybrid

[Figure 21]

[Figure 22]

Redbox Control

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

Style Transfer

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

Extract Reference

- 1 INTRODUCTION

Recent advances in diffusion models (Song et al., 2020; Rombach et al., 2022; Lipman et al., 2022) have significantly improved Text-to-Image (T2I) generation (Li et al., 2024; Lin et al., 2025; Wu et al., 2025a; Labs et al., 2025; Yan et al., 2025a), enabling high-quality and diverse image synthesis. Building on this, diffusion models are increasingly being extended to image editing, which requires precise instruction-following and fine-grained control while preserving unedited content.

Image editing methods include workflow-based (Rombach et al., 2022; Zhang et al., 2023b; Ye et al., 2023) and instruction-based approaches. While recent instruction-based models offer the convenience of unified, prompt-driven editing within a single architecture (Brooks et al., 2023; Wu et al., 2025a; Labs et al., 2025; Lin et al., 2025; Deng et al., 2025; Liu et al., 2025b; OpenAI, 2025), they often struggle with generalization and controllability. These shortcomings stem from the inherent limitations of the Supervised Fine-Tuning (SFT) paradigm. The SFT objective tends to shortcut learning, causing models to ignore complex instructions and revert to merely reconstructing the input (Labs et al., 2025; Wu et al., 2025a; Liu et al., 2025b). Furthermore, its reliance on largescale but less diverse datasets makes models prone to overfitting, compromising their instruction fidelity across diverse tasks. To overcome these issues and better align models with human intent, post-training alignment via Reinforcement Learning (RL) has emerged as a promising direction.

Motivated by recent advances in applying RL to language models (Shao et al., 2024; Yu et al., 2025b), several works (Liu et al., 2025a; Xue et al., 2025b) have explored integrating RL with diffusion models. However, recent studies (Zheng et al., 2025; Xue et al., 2025a) highlight that policy optimization methods employing likelihood estimation can introduce systematic bias and limit solver flexibility. Moreover, these methods (Liu et al., 2025a; Xue et al., 2025b) relying on first-order SDE samplers force a trade-off between trajectory diversity and generation quality. These challenges are particularly acute in image editing, where both high-fidelity generation and diverse exploration are crucial for achieving satisfactory results.

Beyond policy optimization, the success of RL depends on a high-quality reward signal. MLLMs are well-suited for the subjective evaluation required in image editing, offering assessments aligned with human intent. Existing MLLM scoring methods can be categorized as logit-based (Wu et al., 2024; Zhang et al., 2024b; Gong et al., 2025; Li et al., 2025; Xu et al., 2023), which use token distribution statistics for interpretability, and sampling-based (Wang et al., 2025b; Xu et al., 2024; Luo et al., 2025; Wu et al., 2025d), which extract scores from generated outputs. While some approaches (Wang et al., 2025b) use Chain-of-Thought (CoT) to improve reward accuracy, they can introduce exposure bias (Huang et al., 2025a) and high computational costs, leading to a skewed reward distribution under logit-based scoring. Alternatively, fine-tuning improves domain-specific scoring, yet demanding carefully designed datasets, which are necessary to avoid bias and catastrophic forgetting (Huang et al., 2025a) and costly for diverse editing tasks. Therefore, an ideal reward mechanism should leverage the powerful visual understanding priors of MLLMs without costly fine-tuning or unreliable complex reasoning.

We advance post-training for editing models along two main directions: (1) adopting a more effective policy optimization method and (2) leveraging pre-trained MLLMs for reliable, low-cost, and low-hallucination reward signals. In this work, we propose an efficient post-training framework, Edit-R1, which provides an integrated solution to both challenges. First, we adopt DiffusionNFT (Zheng et al., 2025) for policy optimization. This method decouples training and sampling, supports black-box solvers, and removes the need for likelihood estimation. Second, we introduce a novel training-free reward model derived from pretrained MLLMs that does not require CoT reasoning. Instead of sampling-based scoring, we apply a logit-based scoring mechanism to compute expected scores directly from token logits, improving interpretability and computational efficiency.

To validate Edit-R1, we apply our method to diverse base models, including UniWorld-V2 (Lin et al., 2025), FLUX.1 Kontext [Dev] (Labs et al., 2025), and Qwen-Image-Edit [2509] (Wu et al., 2025a). It elevates FLUX.1-Kontext [Dev] surpasses its stronger Pro version and sets a new open source state-of-the-art when applied to Qwen-Image-Edit [2509], achieving scores of 4.48 on ImgEdit and 7.79 on GEdit-Bench. Moreover, UniWorld-V2 yields state-of-the-art performance of 4.49 and 7.83, respectively, outperforming prominent closed-source models. These results are shown in Figure 1 and underscore that our model framework consistently unlocks the potential within various models, showcasing strong generalization.

[Figure 33]

(1) Sampling (2) MLLM Scoring

× N 0.6 0 1 2 3 4 5

Solver

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

Velocity Predictor

MLLM Reward Model

️️

Original Edited

Base Prompt Task Prompt

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

Please evaluate the edited image based on the following editing ...

Replace Adjust Remove Extract

Text Encoder

...

Change the sky to stormy

(3) Diffusion Negative-aware Finetuning

[Figure 42]

Forward Process

+

[Figure 43]

[Figure 44]

|| +(  | ) −  ||22

[Figure 45]

[Figure 46]

+ 

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

Group Filtering

1 −

+ 푚  

[Figure 51]

1 −

{  }

{  }

-

|| −(  | ) −  ||22

Optimality Rewards

1 −

Drop low std group.

Figure 2: Overview of the Edit-R1 pipeline. Our framework consists of three parts: 1) We employ the DPM-Solver (Lu et al., 2022) to perform a rapid rollout, generating a group of candidate images from the policy. 2) We use implicit feedback from MLLM to score the image editing effect and provide rewards. The scoring instructions include both a basic instruction for general editing requirements and a task instruction designed for fine-grained scoring based on the specific editing task type. 3) We fine-tune the velocity predictor using DiffusionNFT (Zheng et al., 2025), enhanced by group filtering method that removes low-variance groups.

Our main contributions are as follows:

- • We propose the Edit-R1 framework, which employs DiffusionNFT and a training-free reward model derived from pretrained MLLMs to fine-tune diffusion models for image editing.
- • We validate that our reward signal offers superior alignment with human preferences, providing reliable, low-cost, and low-hallucination reward signals that stabilize training.
- • Experimental results show that our method significantly improves the performance of UniWorldV2, Qwen-Image-Edit, and FLUX.1-Kontext across diverse editing benchmarks.

- 2 METHODOLOGY

- 2.1 PRELIMINARY

Flow Matching. Given a data sample x0 ∼ X0 with a corresponding condition c (e.g., a class label or text embedding). from the true distribution and a gaussian noise sample x1 ∼ X1, Rectified Flow (Liu et al., 2022; Lipman et al., 2022) defines an interpolation noised sample xt as,

xt = (1 − t)x0 + tx1, (1)

where t ∈ [0,1]. Given c as the text embedding, a neural network vθ(xt,t,c) is trained to approximate the target velocity field v = x1 − x0 by minimizing the flow matching objective:

0∼X0,x1∼X1[||v − vθ(xt,t,c)||22]. (2) Inference is performed by solving a deterministic ODE for the forward process:

LFM(θ) = Et,x

dxt = vθ(xt,t,c)dt. (3)

Diffusion Negative-aware Finetuning (DiffusionNFT). Unlike RL algorithms (Shao et al., 2024; Liu et al., 2025a; Xue et al., 2025b), built upon the policy gradient framework, DiffusionNFT (Zheng et al., 2025) performs policy optimization directly on the forward diffusion process via the flow

matching objective. The method leverages a reward signal r(x0,c) to define a contrastive loss that steers the model’s velocity predictor, vθ, toward the high-reward policy and away from the lowreward one. The core policy optimization loss is defined as:

#### L(θ) = Ec,πold(x0|c),t r ∥vθ+(xt,c,t) − v∥22 + (1 − r)∥vθ−(xt,c,t) − v∥22 , (4)

where v is the target velocity field. The implicit positive and negative policies vθ+ and vθ− are combinations of the old policy vold and the training policy vθ, weighted by a hyperparameter β:

vθ+(xt,c,t) := (1 − β)vold(xt,c,t) + β vθ(xt,c,t), (5) vθ−(xt,c,t) := (1 + β)vold(xt,c,t) − β vθ(xt,c,t). (6)

The optimality probability r ∈ [0,1] is transformed from the unconstrained raw reward signal rraw:

rraw(x0,c) − Eπold(·|c)rraw(x0,c) Zc

- 1

- 2

- 1

- 2

,−1,1 , (7) where Zc > 0 is a normalizing factor, such as the global std of the rewards.

clip

+

r(x0,c) :=

- 2.2 TRAINING-FREE MLLM SCORING

Our approach leverages a pretrained MLLM as a training-free reward model to evaluate the editing accuracy. An editing task is defined by an input sequence X = (Ioriginal,Iedited,Tinstruction), containing the original image, the edited image, and the text instruction. The MLLM’s response generation is modeled as a sequential token-by-token process. Let R<n = (r0,r1,...,rn−1) . The generation of the next token rn+1 is conditioned on the previous tokens:

#### p(rn|X,R<n) = softmax(D(X,R<n)). (8)

Here, D denotes an MLLM, and its output is the logit vector corresponding to the final token in the sequence. We explore the MLLM’s evaluation framework along two dimensions: Chain-of-Thought (CoT) versus non-CoT Scoring, and Sampling-based(discrete) versus logit-based(continuous) Scoring.

CoT vs. non-CoT This dimension explores whether the MLLM should generate explanatory reasoning before providing the final score. In non-CoT Scoring, the MLLM directly produces a score without reasoning, resulting in a response length n = 1. Conversely, CoT Scoring requires the MLLM to generate a CoT reasoning before giving a score, which leads to a response length n > 1.

Sampling-based vs. logit-based This dimension explores how the MLLM’s output is converted into a reward signal. First, the MLLM D generates a textual response R based on a predefined template. Sampling-based Scoring extracts explicit numerical scores from R via deterministic rules.

sdis(X) = Parse(R). (9) This method is simple but yields sparse signals, disregarding the model uncertainty when scoring. logit-based Scoring provides a more fine-grained reward, which is calculated as the expected numerical value of the score tokens:

w(r) · p(rn = r|X,R<n), (10)

scon(X) =

r∈M

where w(r) is the numerical value of the token r, and M denotes the set of tokens used for scoring. This score captures the model’s confidence distribution across scores. Then we normalize scores to the range [0,1]:

s(X) − minr∈M w(r) maxr∈M w(r) − minr∈M w(r)

. (11)

s(X) =

In Edit-R1, we utilize the non-CoT and logit-based scoring method and M = {0,1,2,3,4,5}. Furthermore, we evaluate it against other scoring mechanisms to validate its effectiveness. These include sample-based methods and CoT variants that prompt the MLLM to reason before scoring. Additionally, we benchmark against a pre-trained reward model from existing work (Wang et al., 2025b). For detailed descriptions of each method, please refer to Appendix B.1.

- 2.3 LOW-STD GROUP FILTERING

[Figure 52]

0.96 0 0.98 1

[Figure 53]

[Figure 54]

Figure 3: An example of “change the car to red” where samples within a group are typically successful, causing noise amplification after normalization.

A potential limitation arises from the normalization operation under conditions of low reward variance. When the probabilities assigned by the MLLM to in-group samples are very similar (e.g. , all above 0.95), the minor differences between them cannot reliably indicate a true quality gap. However, in low-variance scenarios, dividing by a standard deviation exaggerates these insignificant scoring differences, as illustrated in Figure 3. The resulting reward signal, which reflects noise rather than true quality, can mislead the training process. Filtering out these noisy groups is crucial for maintaining training stability.

Therefore, we aim to filter out groups with high means

and low variance in raw rewards. Specifically, we introduce two hyperparameters, τµ and τσ, which represent the thresholds for the mean and variance, respectively. During training, gradients from

groups whose mean reward exceeds τµ and whose variance falls below τσ are discarded and do not contribute to the optimization process.

- 2.4 PIPELINE OF EDIT-R1

To enhance the image editing model, we leverage DiffusionNFT (Zheng et al., 2025) and adopt reward signals from an MLLM. This approach makes the reward signal universally applicable to any editing tasks, generating stable rewards from the same distribution for policy optimization while eliminating the reliance on domain-specific reward models.

As illustrated in Figure 2, the process consists of three main parts: sampling, MLLM scoring, and diffusion negative-aware finetuning, which progressively align the model with the optimal policy.

- Part 1: Sampling Benefiting from the decoupling of policy optimizing and data sampling, DiffusionNFT enables the full utilization of any black-box solvers throughout sampling. Therefore, we specifically employ the DPM-Solver (Lu et al., 2022) to perform a rapid rollout for a given source image and an edit instruction, generating a set of G images {xi0}Gi=1 sampled from the policy πold.
- Part 2: MLLM scoring We evaluate the generated image group {xi0}Gi=1 based on implicit feedback from the MLLM to measure alignment with the editing instruction and overall image quality. Conditioned on original image, edited image, and evaluation prompt, the MLLM generates a series

of raw reward scores {si}Gi=1 for {xi0}Gi=1. To facilitate fine-grained scoring, the evaluation prompt is structured into two components: a base prompt, which outlines fundamental editing requirements

and instructions, and a task prompt, which is specifically tailored to the type of editing task.

- Part 3: DiffusionNFT The raw MLLM scores {si}Gi=1 are transformed into optimality rewards

{ri}Gi=1 through group computation. These rewards are then used to update the policy model vθ using the DiffusionNFT objective, as defined in Equation 4. This process guides the model’s velocity

predictor towards high-reward outcomes while moving it away from low-reward ones, effectively fine-tuning the model to better adhere to user instructions and produce higher-quality edits.

- 3 EXPERIMENTS

- 3.1 DATASET

We curate a dataset comprising 27,572 instruction-based editing samples in total (Figure 5), which are sourced from LAION (Schuhmann et al., 2022), LexArt (Zhao et al., 2025), and UniWorldV1 (Lin et al., 2025). To enhance task diversity, we incorporate additional text-editing and red-box control tasks, resulting in nine distinct task types. Leveraging an online learning paradigm, our method operates solely on the original images and their corresponding editing instructions, which eliminates the need for high-quality edited result images.

Change the animal's fur color to a light brown shade.

Make the person lift his head slightly.

Remove the dog from the image.

Replace the text SHEFFIELD with EDINBURGH

Extract the yellow necktie worn in the image.

Replace the person in the image with a vintage gramophone

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

Original

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

FLUX-Kontext [dev]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

UniWorld-FLUXKontext [dev]

Remove the armchair with wooden armrests and checkered upholstery from the image.

Extract the road bicycle leaning against the gate in the image.

Change the text ONTARIO to ONTARO

Add a bicycle leaning against the fence on the right side of the path.

Raise the person's left arm.

Change the vehicle color to red.

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

Original

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

Qwen-Image-Edit [2509]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

UniWorld-QwenImge-Edit [2509]

### Figure 4: Qualitative comparison of basic editing capabilities before and after policy optimization.. Basic editing refers to single-step modifications applied to an image.

For the LAION subset, we utilize the existing object annotations and bounding boxes provided by ImgEdit (Ye et al., 2025b). The preprocessing pipeline includes: 1) Filtering out bounding boxes that are either too small or excessively large. 2) Using the Qwen2.5-VL-32B model to assess the rationality of the editing instructions. For the Text Edit task, we build upon the LexArt subset by randomly altering characters in words to generate training samples. In the Redbox Control task, we extract a subset from the processed LAION data, draw red bounding boxes around target objects, and generate three types of edit instructions: Adjust, Remove, and Replace. For the Reference and Extract tasks, we employ high-quality tryon data from UniWorld-V1. Due to the limited diversity in this dataset, we use only 600 samples for each of these two tasks.

Replace Adjust Remove Background Hybrid Action Text Ddit Redbox Control Reference Extract

| |
|---|

| |
|---|

| |
|---|

2.2%2.2%

| |
|---|

6.5%

21.1%

| |
|---|

| |
|---|

7.3%

| |
|---|

| |
|---|

| |
|---|

7.3%

Total 27572

9.1%

18.5%

10.9%

15.1%

Figure 5: Data Composition Overview. Our dataset comprises 9 tasks: Replace, Adjust, Remove, Background, Hybrid, Action, Text Edit, Redbox Control, Reference, and Extract.

- Table 1: Quantitative comparison results on ImgEdit (Ye et al., 2025b). We use GPT4.1 for evaluation. Bold indicates the best performance.

Model Add Adjust Extract Replace Remove Background Style Hybrid Action Overall ↑

MagicBrush (Zhang et al., 2023a) 2.84 1.58 1.51 1.97 1.58 1.75 2.38 1.62 1.22 1.90 Instruct-Pix2Pix (Brooks et al., 2023) 2.45 1.83 1.44 2.01 1.50 1.44 3.55 1.20 1.46 1.88 AnyEdit (Yu et al., 2025a) 3.18 2.95 1.88 2.47 2.23 2.24 2.85 1.56 2.65 2.45 UltraEdit (Zhao et al., 2024) 3.44 2.81 2.13 2.96 1.45 2.83 3.76 1.91 2.98 2.70 OmniGen (Xiao et al., 2025) 3.47 3.04 1.71 2.94 2.43 3.21 4.19 2.24 3.38 2.96 ICEdit (Zhang et al., 2025b) 3.58 3.39 1.73 3.15 2.93 3.08 3.84 2.04 3.68 3.05 Step1X-Edit (Liu et al., 2025b) 3.88 3.14 1.76 3.40 2.41 3.16 4.63 2.64 2.52 3.06 BAGEL (Deng et al., 2025) 3.56 3.31 1.70 3.3 2.62 3.24 4.49 2.38 4.17 3.20 UniWorld-V1 (Lin et al., 2025) 3.82 3.64 2.27 3.47 3.24 2.99 4.21 2.96 2.74 3.26 OmniGen2 (Wu et al., 2025b) 3.57 3.06 1.77 3.74 3.20 3.57 4.81 2.52 4.68 3.44 FLUX.1 Kontext [Pro] (Labs et al., 2025) 4.25 4.15 2.35 4.56 3.57 4.26 4.57 3.68 4.63 4.00 GPT-Image-1 [High] (OpenAI, 2025) 4.61 4.33 2.90 4.35 3.66 4.57 4.93 3.96 4.89 4.20 Qwen-Image-Edit (Wu et al., 2025a) 4.38 4.16 3.43 4.66 4.14 4.38 4.81 3.82 4.69 4.27

FLUX.1 Kontext [Dev] (Labs et al., 2025) 4.12 3.80 2.04 4.22 3.09 3.97 4.51 3.35 4.25 3.71 UniWorld-FLUX.1-Kontext 4.19 4.20 2.43 4.32 3.91 4.08 4.68 3.72 4.63 4.02 vs. Baseline +0.07 +0.40 +0.39 +0.10 +0.82 +0.11 +0.17 +0.37 +0.38 +0.31

Qwen-Image Edit [2509] (Wu et al., 2025a) 4.32 4.36 4.04 4.64 4.52 4.37 4.84 3.39 4.71 4.35 UniWorld-Qwen-Image-Edit 4.34 4.40 4.37 4.62 4.65 4.31 4.89 3.90 4.80 4.48 vs. Baseline +0.02 +0.04 +0.33 -0.02 +0.13 -0.06 +0.05 +0.51 0.09 +0.13

UniWorld-V2 4.29 4.44 4.32 4.69 4.72 4.41 4.91 3.83 4.83 4.49

- 3.2 EXPERIMENTAL SETUP

To evaluate the effectiveness of our approach, we conduct experiments from two perspectives: 1) The alignment between different MLLM scoring methods and human judgments, and 2) The performance improvement of the editing model after post-training with our method.

Training We use FLUX.1-Kontext [Dev] (Labs et al., 2025), Qwen-Image-Edit [2509] (Wu et al.,

- 2025a) and UniWorld-V2 as our base models. For training, we allocate 3 nodes to FLUX.1-Kontext [Dev], 6 nodes to Qwen-Image-Edit [2509] and 9 nodes to UniWorld-V2, with each node containing 8 A100 GPUs. We perform MLLM scoring on a single node using vLLM (Kwon et al., 2023). To optimize GPU memory utilization, we employ Fully Sharded Data Parallelism (FSDP) for the text encoder and gradient checkpointing when training Qwen-Image-Edit [2509] and UniWorld-V2.

Evaluation For quantitative evaluation, we employ two comprehensive benchmarks: ImgEdit (Ye et al., 2025b), which unifies multiple specialized tasks into a common framework for comprehensive model comparison, and GEdit-Bench (Liu et al., 2025b), which assesses general-purpose image editing through diverse natural language instructions.

- 3.3 MAIN RESULTS

We evaluate these models on the ImgEdit and GEdit-Bench benchmarks to assess their editing capabilities and generalization. The quantitative results are presented in Table 1 and Table 2, respectively, and a qualitative comparison is shown in Figure 4.

Our method unlocks the model’s potential and significantly improves its performance. As shown on the ImgEdit benchmark in Table 1, our method substantially enhances the performance of all base models. For FLUX.1-Kontext [Dev], the overall score improves significantly from 3.71 to 4.02, outperforming the stronger Pro version (4.00). Similarly, when applied to Qwen-ImageEdit [2509], our method boosts its score from 4.35 to an impressive 4.48, achieving state-of-the-art performance among open-source models and surpassing top-tier closed-source models like GPTImage-1. Beyond the gains in the overall score, a dramatic performance surge is observed in the ‘Adjust’, ‘Extract’, and ‘Remove‘ dimensions for UniWorld-FLUX.1-Kontext and in ‘Extract’ and ‘Hybrid’ dimensions for UniWorld-Qwen-Image-Edit. Moreover, UniWorld-V2 achieves the best performance. This phenomenon indicates that our method can unlock and significantly improve the previously underdeveloped potential within the base models.

Our method exhibits robust generalization capabilities on the out-of-domain dataset. On the out-of-domain GEdit-Bench (Table 2), Edit-R1 demonstrates strong generalization for three models. It enhances the FLUX.1-Kontext [Dev] model’s overall score from 6.00 to 6.74, yielding a performance that surpasses the Pro version (6.56). For the Qwen-Image model, the score is increased

- Table 2: Quantitative comparison on GEdit-Bench (Liu et al., 2025b). Bold indicates the best performance.

Model GEdit-Bench-EN ↑ G SC G PQ G O

Instruct-Pix2Pix 3.58 5.49 3.68 AnyEdit 3.18 5.82 3.21 MagicBrush 4.68 5.66 4.52 UniWorld-V1 4.93 7.43 4.85 OmniGen 5.96 5.89 5.06 FLUX.1-Kontext [Dev] 6.52 7.38 6.00 OmniGen2 7.16 6.77 6.41 Gemini 2.0 6.73 6.61 6.32 BAGEL 7.36 6.83 6.52 FLUX.1-Kontext [Pro] 7.02 7.60 6.56 Step1X-Edit 7.66 7.35 6.97 UniPic2 - - 7.10 GPT-Image-1 [High] 7.85 7.62 7.53 Qwen-Image-Edit 8.00 7.86 7.56

FLUX.1-Kontext [Dev] 6.52 7.38 6.00 UniWorld-FLUX.1-Kontext 7.28 7.49 6.74 vs. Baseline +0.76 +0.11 +0.74

Qwen-Image-Edit [2509] 8.15 7.86 7.54 UniWorld-Qwen-Image-Edit 8.36 7.87 7.76 vs. Baseline +0.21 +0.01 +0.22

UniWorld-V2 8.39 8.02 7.83

K-R1

K-Pro

K-Dev

| |
|---|

| |
|---|

100

17.5%

20.1%

22.7%

80

38.2%

38.8%

60

39.4%

40

44.3%

41.1%

20

37.9%

0

Alignment Quality Overall

Q-R1

Q-2509

Q-Base

| |
|---|

| |
|---|

100

27.1%

29.4%

31.6%

80

60

34.5%

34.8%

35.0%

40

20

38.4%

35.9%

33.4%

0

Alignment Quality Overall

Figure 6: User study for Kontext-based (up) and Qwen-based (bottom) model.

from 7.54 to 7.76. Meanwhile, UniWorld-V2 establishes a new state-of-the-art on this benchmark by outperforming all listed models, including Qwen-Image-Edit (7.56) and GPT-Image-1 (7.53). This result confirms that our method effectively preserves and enhances core editing capabilities on unseen data distributions, showcasing strong generalization.

Our method proves its effectiveness in human preference evaluations. For a comprehensive evaluation, we conducted a human preference study for both the FLUX.1 and Qwen series, where participants compared our finetuned model with its base models and the more powerful version. They were asked to select the best result across two dimensions: instruction alignment and image quality. As detailed in Figure 6, users prefer the UniWorld-FLUX.1-Kontext over FLUX.1-Kontext [Dev] across all criteria. Furthermore, it demonstrates stronger editing capabilities when compared to the more powerful official versions, FLUX.1-Kontext [Pro]. Overall, the UniWorld-FLUX.1Kontext gains more likes due to its superior instruction-following ability, even though the official models are slightly superior in image quality. This confirms that our method effectively steers the model to generate outputs that better align with human preferences.

4.02

4.00

3.90

ImgEdit

| |
|---|

| |
|---|

NFT

| |
|---|

3.80

FlowGRPO

FlowGRPO+Local Std

3.70

0 100 200 300 400 500 Step

- Figure 7: Results of different policy optimization methods on FLUX.1-Kontext [Dev].

Table 3: Ablation study of our core components using Qwen-Image-Edit [2509] on GEdit-Bench.

Model GEdit-Bench ↑ Qwen-Image-Edit [2509] 7.54 + NFT (7B) 7.66 + 32B 7.74 + Group Filtering 7.76

- 3.4 ABLATION STUDY

We conducted ablation studies to validate our core components. As presented in Figure 7, we employ DiffusionNFT as the policy optimization method on FLUX.1 Kontext [Dev]. It achieves superior performance on the ImgEdit benchmark over baselines, including Flow-GRPO and its variant using local std. Moreover, as shown in Table 3, applying DiffusionNFT to the Qwen-Image-Edit [2509]

baseline model significantly raises its score on GEdit-Bench from 7.54 to 7.72. Introducing the group filtering mechanism further increases the score to 7.76.

- 3.5 ANALYSIS

[Figure 94]

[Figure 95]

[Figure 96]

| |3B 7B 32B<br><br>| | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

0.90

3B

0.85

0.80

Reward

[Figure 97]

[Figure 98]

[Figure 99]

0.75

0.70

7B

0.65

0.60

0 50 100 150 200 250

Step

Training Progress

- Figure 8: Reward hacking phenomenon observed in the 3B reward model (Left) and Training reward dynamics across varying reward model scales (Right).

S_L

Material Alter

Table 4: Pairwise accuracy of different reward methods against human preferences

UR

S_L+CoT S_S+CoT Y/N_L

Motion

Color

S_S

75%

### Reward Method Accuracy (%)

50%

Score Sampling (S S) 60.82 Yes/No Logit (Y/N L) 67.01 Score Sampling + CoT 62.37 Score Logit + CoT 63.40 UnifiedReward (UR) (Wang et al., 2025b)

25%

Style

Background

Add

Replace

65.46 Score Logit (Ours) 74.74

Remove

Figure 9: Performance comparison across different editing tasks.

Human Alignment. To validate our choice of reward mechanism, we evaluate the alignment between different scoring methods and human judgments. The results indicate that our adopted logitbased method achieves the highest correlation with human preferences among all evaluated reward mechanisms. As detailed in Table 4, this method achieves an overall pairwise accuracy of 74.74%, significantly outperforming other methods. Furthermore, the results in Figure 9 demonstrate that superior alignment is consistent across diverse editing tasks. We provide a detailed analysis of these findings in Appendix B, including score distributions and the detailed experimental protocol.

Reward Model Scaling. To evaluate the effect of reward model scaling on policy model performance, we fine-tuned Qwen-Image-Edit for the same number of steps using reward models of varying parameter scales for a fair comparison. As shown in Table 3, with an increase in the scale of the reward model, the overall score of the policy model improves, demonstrating that scaling the reward model contributes to continuous performance improvement.

Reward Hacking and Reward Variance. As illustrated in Figure 8 (Left), the policy model finetuned on 3B model exhibits significant reward hacking and its edit results deviate from the source image. In contrast, the model fine-tuned on larger 7B model alleviates this issue. For further investigation, we analyze the training reward curves and attribute the phenomenon to the variance of reward scores. As depicted in Figure 8 (Right), we visualize the smoothed reward trajectory (solid

line), raw reward fluctuations (shaded line), and the reward variance (bubble size) as an indicator of exploration intensity.

Our observations are as follows: i) Reward Hacking in Smaller Models: Smaller reward model, such as 3B and 7B, exhibit reward hacking, as their reward variance rapidly collapses early in training, indicating a premature halt to effective exploration. ii) Sustained Exploration in Larger Models: In contrast, the 32B model maintains high reward variance throughout training, demonstrating sustained exploration capability which enables the discovery of superior solutions even in later stages. This phenomenon is also analyzed in another study (Wu et al., 2025c). These dynamics suggest that scaling reward models can effectively mitigate reward hacking and maintain robust exploration.

- 4 RELATED WORK

- 4.1 IMAGE EDITING

The advent of diffusion models has marked a pivotal moment in Text-to-Image (T2I) generation (Song et al., 2020; Rombach et al., 2022; Lipman et al., 2022; Liu et al., 2025c; Yan et al., 2025b). Image editing presents a more constrained challenge: altering specific attributes while preserving unedited regions. Early methods like SDEdit (Meng et al., 2021) offered global stylistic control but lacked spatial precision. To address this, inversion-based techniques (Hertz et al., 2022) were developed to reconstruct an image from its latent representation, with methods like Null-text Inversion (Mokady et al., 2023) optimizing the process, though they can be computationally intensive and may introduce artifacts (Huang et al., 2025b). Other methods focus on adding explicit conditional control, such as ControlNet (Zhang et al., 2023b) for spatial guidance and IP-Adapter (Ye et al.,

- 2023) for image-based prompting. Concurrently, training-based fine-tuning approaches (Brooks et al., 2023) adapted models for editing but faced generalization issues. Building on these foundations, a new wave of powerful instruction-tuned models such as ICEdit (Zhang et al., 2025b) and Step1X-Edit (Liu et al., 2025b) has emerged. These are increasingly integrated with general-purpose MLLMs, leading to highly capable systems like BAGEL (Deng et al., 2025), Qwen-Image (Wu et al., 2025a), and GPT-Image-1 (OpenAI, 2025), while foundational architectures also evolve with frameworks like flow matching in FLUX Kontext (Labs et al., 2025).

4.2 REINFORCEMENT LEARNING IN GENERATIVE MODELS

Reinforcement Learning From Human Feedback (RLHF) (Ouyang et al., 2022) has become the dominant paradigm for aligning LLMs to be more helpful (Hu et al., 2022; Shao et al., 2024), honest (Gao et al., 2024), and harmless (Yang et al., 2025). Inspired by this success, researchers have adapted the RL framework to T2I models (Black et al., 2023), typically by training a reward model (RM) on general human preferences (Xu et al., 2024) or specific prompt-image alignment scores (Xu et al., 2023). However, this “train-an-RM-then-RL” pipeline is suboptimal for image editing, as RMs are difficult to craft (Ye et al., 2025a; Miao et al., 2024). Alternatives like Direct Preference Optimization (DPO) (Rafailov et al., 2023) remove the explicit RM but rely on static, offline data, which provides a less dynamic signal than required for iterative visual refinement. Similarly, using powerful MLLMs to provide discrete scores (Gong et al., 2025; Niu et al., 2025) yields a coarse signal that fails to capture the continuous nature of visual quality. More advanced algorithms like GRPO (Shao et al., 2024) have also shown promise in aligning both diffusion and flow matching models, as seen in FlowGRPO (Liu et al., 2025a), which was first utilized in Skywork UniPic 2.0 (Wei et al., 2025) for image editing, and DanceGRPO (Xue et al., 2025b), but can still be exploited via reward hacking (Wang et al., 2025a). Differing from these policy gradient approaches, DiffusionNFT (Zheng et al., 2025) presents an online RL paradigm that directly optimizes the model on the forward process via a negative-aware fine-tuning objective.

4.3 MLLMS AS EVALUATORS AND REWARD MODELS The emergence of powerful MLLMs has established the “MLLM-as-a-Judge” paradigm (Chen et al.,

- 2024), demonstrating a high correlation with human judgments (Zhang et al., 2025a). This has motivated a shift from passive evaluation to using MLLMs as an active reward to optimize generative models (Gong et al., 2025; Xu et al., 2023; Jin et al., 2025; Niu et al., 2025). However, converting

MLLM evaluations into an effective reward signal for image editing presents challenges. A straightforward approach of using discrete scores (Gong et al., 2025) provides a sparse and coarse signal, ill-suited for the continuous and subtle nature of visual improvements. To obtain finer-grained reward signals, logit-based scoring methods generate rewards by computing the expected value of the token distribution from model outputs (Wu et al., 2024; Zhang et al., 2024b; Li et al., 2025). Another strategy involves learning from large, static pairwise preference datasets (Xu et al., 2023), which are often used in alignment algorithms like DPO adapted for T2I models (Black et al., 2023). This offline method decouples the feedback from the live generation process, potentially failing to cover the vast distribution of possible edits and lacking the real-time, iterative guidance needed for optimization (Zhang et al., 2024a). The goal is to develop a feedback mechanism that can effectively guide editing models that are already leveraging MLLMs for planning and reasoning (Liu et al., 2025c; Yang et al., 2024). Another feedback mechanism to guide editing models involves leveraging MLLMs for planning and reasoning (Liu et al., 2025c; Yang et al., 2024), offering more direct and interpretable guidance than a simple feedback score.

- 5 CONCLUSION

In this paper, we introduce Edit-R1, a novel post-training framework designed to overcome generalization limitations in instruction-based image editing models. Our core innovation is using an MLLM as a training-free reward model, which provides fine-grained, continuous feedback directly from its output logits, combined with the efficient DiffusionNFT, a likelihood-free policy optimization method consistent with the flow matching forward process. Extensive experiments demonstrate that our framework achieves state-of-the-art performance on the ImgEdit and GEdit-Bench by significantly boosting various base models, including UniWorld-V2, FLUX.1-Kontext, and Qwen-ImageEdit. Our analysis confirms that the MLLM-derived reward signal exhibits a high correlation with human preferences, effectively guiding the model towards higher-quality outputs while mitigating reward hacking.

- 6 CONTRIBUTORS

Core Contributors: Zongjian Li, Zheyuan Liu, Qihui Zhang, Bin Lin Contributors: Feize Wu, Shenghai Yuan, Zhiyuan Yan, Yang Ye, Wangbo Yu, Yuwei Niu, Shaodong Wang, Xinhua Cheng Corresponding Author: Li Yuan {zongjianli25@stu., yuanli-ece@}pku.edu.cn

REFERENCES

Kevin Black, Michael Janner, Yilun Du, Ilya Kostrikov, and Sergey Levine. Training diffusion models with reinforcement learning. arXiv preprint arXiv:2305.13301, 2023.

Tim Brooks, Aleksander Holynski, and Alexei A Efros. Instructpix2pix: Learning to follow image editing instructions. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 18392–18402, 2023.

Dongping Chen, Ruoxi Chen, Shilin Zhang, Yaochen Wang, Yinuo Liu, Huichi Zhou, Qihui Zhang, Yao Wan, Pan Zhou, and Lichao Sun. Mllm-as-a-judge: Assessing multimodal llm-as-a-judge with vision-language benchmark. In Forty-first International Conference on Machine Learning, 2024.

Chaorui Deng, Deyao Zhu, Kunchang Li, Chenhui Gou, Feng Li, Zeyu Wang, Shu Zhong, Weihao Yu, Xiaonan Nie, Ziang Song, et al. Emerging properties in unified multimodal pretraining. arXiv preprint arXiv:2505.14683, 2025.

Chujie Gao, Siyuan Wu, Yue Huang, Dongping Chen, Qihui Zhang, Zhengyan Fu, Yao Wan, Lichao Sun, and Xiangliang Zhang. Honestllm: Toward an honest and helpful large language model. arXiv preprint arXiv:2406.00380, 2024.

Yuan Gong, Xionghui Wang, Jie Wu, Shiyin Wang, Yitong Wang, and Xinglong Wu. Onereward: Unified mask-guided image generation via multi-task human preference learning. arXiv preprint arXiv:2508.21066, 2025.

Amir Hertz, Ron Mokady, Jay Tenenbaum, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Prompt-to-prompt image editing with cross attention control. arXiv preprint arXiv:2208.01626, 2022.

Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al. Lora: Low-rank adaptation of large language models. ICLR, 1(2):3, 2022.

Lei Huang, Weijiang Yu, Weitao Ma, Weihong Zhong, Zhangyin Feng, Haotian Wang, Qianglong Chen, Weihua Peng, Xiaocheng Feng, Bing Qin, et al. A survey on hallucination in large language models: Principles, taxonomy, challenges, and open questions. ACM Transactions on Information Systems, 43(2):1–55, 2025a.

Yi Huang, Jiancheng Huang, Yifan Liu, Mingfu Yan, Jiaxi Lv, Jianzhuang Liu, Wei Xiong, He Zhang, Liangliang Cao, and Shifeng Chen. Diffusion model-based image editing: A survey. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2025b.

Weiyang Jin, Yuwei Niu, Jiaqi Liao, Chengqi Duan, Aoxue Li, Shenghua Gao, and Xihui Liu. Srum: Fine-grained self-rewarding for unified multimodal models. arXiv preprint arXiv:2510.12784, 2025.

Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with pagedattention. In Proceedings of the 29th symposium on operating systems principles, pp. 611–626, 2023.

Black Forest Labs, Stephen Batifol, Andreas Blattmann, Frederic Boesel, Saksham Consul, Cyril Diagne, Tim Dockhorn, Jack English, Zion English, Patrick Esser, Sumith Kulal, Kyle Lacey, Yam Levi, Cheng Li, Dominik Lorenz, Jonas M¨uller, Dustin Podell, Robin Rombach, Harry Saini, Axel Sauer, and Luke Smith. Flux.1 kontext: Flow matching for in-context image generation and editing in latent space, 2025. URL https://arxiv.org/abs/2506.15742.

Yi-Chen Li, Tian Xu, Yang Yu, Xuqin Zhang, Xiong-Hui Chen, Zhongxiang Ling, Ningjing Chao, Lei Yuan, and Zhi-Hua Zhou. Generalist reward models: Found inside large language models. arXiv preprint arXiv:2506.23235, 2025.

Zhimin Li, Jianwei Zhang, Qin Lin, Jiangfeng Xiong, Yanxin Long, Xinchi Deng, Yingfang Zhang, Xingchao Liu, Minbin Huang, Zedong Xiao, et al. Hunyuan-dit: A powerful multi-resolution diffusion transformer with fine-grained chinese understanding. arXiv preprint arXiv:2405.08748, 2024.

Bin Lin, Zongjian Li, Xinhua Cheng, Yuwei Niu, Yang Ye, Xianyi He, Shenghai Yuan, Wangbo Yu, Shaodong Wang, Yunyang Ge, et al. Uniworld: High-resolution semantic encoders for unified visual understanding and generation. arXiv preprint arXiv:2506.03147, 2025.

Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022.

Jie Liu, Gongye Liu, Jiajun Liang, Yangguang Li, Jiaheng Liu, Xintao Wang, Pengfei Wan, Di Zhang, and Wanli Ouyang. Flow-grpo: Training flow matching models via online rl. arXiv preprint arXiv:2505.05470, 2025a.

Shiyu Liu, Yucheng Han, Peng Xing, Fukun Yin, Rui Wang, Wei Cheng, Jiaqi Liao, Yingming Wang, Honghao Fu, Chunrui Han, et al. Step1x-edit: A practical framework for general image editing. arXiv preprint arXiv:2504.17761, 2025b.

Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. The International Conference on Learning Representations, 2022.

Zheyuan Liu, Munan Ning, Qihui Zhang, Shuo Yang, Zhongrui Wang, Yiwei Yang, Xianzhe Xu, Yibing Song, Weihua Chen, Fan Wang, et al. Cot-lized diffusion: Let’s reinforce t2i generation step-by-step. arXiv preprint arXiv:2507.04451, 2025c.

Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu. Dpm-solver: A fast ode solver for diffusion probabilistic model sampling in around 10 steps. Advances in neural information processing systems, 35:5775–5787, 2022.

Xin Luo, Jiahao Wang, Chenyuan Wu, Shitao Xiao, Xiyan Jiang, Defu Lian, Jiajun Zhang, Dong Liu, et al. Editscore: Unlocking online rl for image editing via high-fidelity reward modeling. arXiv preprint arXiv:2509.23909, 2025.

Chenlin Meng, Yutong He, Yang Song, Jiaming Song, Jiajun Wu, Jun-Yan Zhu, and Stefano Ermon. Sdedit: Guided image synthesis and editing with stochastic differential equations. arXiv preprint arXiv:2108.01073, 2021.

Zichen Miao, Jiang Wang, Ze Wang, Zhengyuan Yang, Lijuan Wang, Qiang Qiu, and Zicheng Liu. Training diffusion models towards diverse image generation with reinforcement learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 10844–10853, 2024.

Ron Mokady, Amir Hertz, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Null-text inversion for editing real images using guided diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 6038–6047, 2023.

Yuwei Niu, Munan Ning, Mengren Zheng, Weiyang Jin, Bin Lin, Peng Jin, Jiaqi Liao, Chaoran Feng, Kunpeng Ning, Bin Zhu, et al. Wise: A world knowledge-informed semantic evaluation for text-to-image generation. arXiv preprint arXiv:2503.07265, 2025.

OpenAI. Image generation API. https://openai.com/index/ image-generation-api/, 2025.

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35: 27730–27744, 2022.

Andrea Pozzi, Alessandro Incremona, Daniele Tessera, and Daniele Toti. Mitigating exposure bias in large language model distillation: an imitation learning approach. Neural Computing and Applications, pp. 1–17, 2025.

Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. Advances in neural information processing systems, 36:53728–53741, 2023.

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. Highresolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 10684–10695, 2022.

Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al. Laion-5b: An open large-scale dataset for training next generation image-text models. Advances in neural information processing systems, 35:25278–25294, 2022.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456, 2020.

Yibin Wang, Zhimin Li, Yuhang Zang, Yujie Zhou, Jiazi Bu, Chunyu Wang, Qinglin Lu, Cheng Jin, and Jiaqi Wang. Pref-grpo: Pairwise preference reward-based grpo for stable text-to-image reinforcement learning. arXiv preprint arXiv:2508.20751, 2025a.

Yibin Wang, Yuhang Zang, Hao Li, Cheng Jin, and Jiaqi Wang. Unified reward model for multimodal understanding and generation. arXiv preprint arXiv:2503.05236, 2025b.

Hongyang Wei, Baixin Xu, Hongbo Liu, Cyrus Wu, Jie Liu, Yi Peng, Peiyu Wang, Zexiang Liu, Jingwen He, Yidan Xietian, et al. Skywork unipic 2.0: Building kontext model with online rl for unified multimodal model. arXiv preprint arXiv:2509.04548, 2025.

Chenfei Wu, Jiahao Li, Jingren Zhou, Junyang Lin, Kaiyuan Gao, Kun Yan, Sheng-ming Yin, Shuai Bai, Xiao Xu, Yilei Chen, et al. Qwen-image technical report. arXiv preprint arXiv:2508.02324, 2025a.

Chenyuan Wu, Pengfei Zheng, Ruiran Yan, Shitao Xiao, Xin Luo, Yueze Wang, Wanli Li, Xiyan Jiang, Yexin Liu, Junjie Zhou, et al. Omnigen2: Exploration to advanced multimodal generation. arXiv preprint arXiv:2506.18871, 2025b.

Haoning Wu, Zicheng Zhang, Erli Zhang, Chaofeng Chen, Liang Liao, Annan Wang, Chunyi Li, Wenxiu Sun, Qiong Yan, Guangtao Zhai, et al. Q-bench: A benchmark for general-purpose foundation models on low-level vision. In ICLR, 2024.

Jie Wu, Yu Gao, Zilyu Ye, Ming Li, Liang Li, Hanzhong Guo, Jie Liu, Zeyue Xue, Xiaoxia Hou, Wei Liu, et al. Rewarddance: Reward scaling in visual generation. arXiv preprint arXiv:2509.08826, 2025c.

Keming Wu, Sicong Jiang, Max Ku, Ping Nie, Minghao Liu, and Wenhu Chen. Editreward: A human-aligned reward model for instruction-guided image editing. arXiv preprint arXiv:2509.26346, 2025d.

Zhaofeng Wu, Robert L. Logan IV, Matt Gardner, and Sameer Singh. Self-correction causes persuasion: The impact of reasoning on factual adherence in llm-based evaluators, 2023.

Shitao Xiao, Yueze Wang, Junjie Zhou, Huaying Yuan, Xingrun Xing, Ruiran Yan, Chaofan Li, Shuting Wang, Tiejun Huang, and Zheng Liu. Omnigen: Unified image generation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 13294–13304, 2025.

Jiazheng Xu, Xiao Liu, Yuchen Wu, Yuxuan Wang, Weiyun Ye, Shihao Geng, Yiren Zhao, Jiaming Li, Cunjian Li, Hang Sun, et al. Imagereward: Learning and evaluating human preferences for text-to-image generation. In Advances in Neural Information Processing Systems, 2023.

Jiazheng Xu, Yu Huang, Jiale Cheng, Yuanming Yang, Jiajun Xu, Yuan Wang, Wenbo Duan, Shen Yang, Qunlin Jin, Shurun Li, et al. Visionreward: Fine-grained multi-dimensional human preference learning for image and video generation. arXiv preprint arXiv:2412.21059, 2024.

Shuchen Xue, Chongjian Ge, Shilong Zhang, Yichen Li, and Zhi-Ming Ma. Advantage weighted matching: Aligning rl with pretraining in diffusion models. arXiv preprint arXiv:2509.25050, 2025a.

Zeyue Xue, Jie Wu, Yu Gao, Fangyuan Kong, Lingting Zhu, Mengzhao Chen, Zhiheng Liu, Wei Liu, Qiushan Guo, Weilin Huang, et al. Dancegrpo: Unleashing grpo on visual generation. arXiv preprint arXiv:2505.07818, 2025b.

Zhiyuan Yan, Kaiqing Lin, Zongjian Li, Junyan Ye, Hui Han, Zhendong Wang, Hao Liu, Bin Lin, Hao Li, Xue Xu, et al. Can understanding and generation truly benefit together–or just coexist? arXiv preprint arXiv:2509.09666, 2025a.

Zhiyuan Yan, Junyan Ye, Weijia Li, Zilong Huang, Shenghai Yuan, Xiangyang He, Kaiqing Lin, Jun He, Conghui He, and Li Yuan. Gpt-imgeval: A comprehensive benchmark for diagnosing gpt4o in image generation. arXiv preprint arXiv:2504.02782, 2025b.

Ling Yang, Zhaochen Yu, Chenlin Meng, Minkai Xu, Stefano Ermon, and Bin Cui. Mastering textto-image diffusion: Recaptioning, planning, and generating with multimodal llms. In Forty-first International Conference on Machine Learning, 2024.

Shuo Yang, Qihui Zhang, Yuyang Liu, Yue Huang, Xiaojun Jia, Kunpeng Ning, Jiayu Yao, Jigang Wang, Hailiang Dai, Yibing Song, et al. Asft: Anchoring safety during llm fine-tuning within narrow safety basin. arXiv preprint arXiv:2506.08473, 2025.

Hu Ye, Jun Zhang, Sibo Liu, Xiao Han, and Wei Yang. Ip-adapter: Text compatible image prompt adapter for text-to-image diffusion models. arXiv preprint arXiv:2308.06721, 2023.

Yang Ye, Tianyu He, Shuo Yang, and Jiang Bian. Reinforcement learning with inverse rewards for world model post-training. arXiv preprint arXiv:2509.23958, 2025a.

Yang Ye, Xianyi He, Zongjian Li, Bin Lin, Shenghai Yuan, Zhiyuan Yan, Bohan Hou, and Li Yuan. Imgedit: A unified image editing dataset and benchmark. arXiv preprint arXiv:2505.20275, 2025b.

Qifan Yu, Wei Chow, Zhongqi Yue, Kaihang Pan, Yang Wu, Xiaoyang Wan, Juncheng Li, Siliang Tang, Hanwang Zhang, and Yueting Zhuang. Anyedit: Mastering unified high-quality image editing for any idea. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 26125–26135, 2025a.

Tianyu Yu, Bo Ji, Shouli Wang, Shu Yao, Zefan Wang, Ganqu Cui, Lifan Yuan, Ning Ding, Yuan Yao, Zhiyuan Liu, Maosong Sun, and Tat-Seng Chua. Rlpr: Extrapolating rlvr to general domains without verifiers, 2025b. URL https://arxiv.org/abs/2506.18254.

Kai Zhang, Lingbo Mo, Wenhu Chen, Huan Sun, and Yu Su. Magicbrush: A manually annotated dataset for instruction-guided image editing. Advances in Neural Information Processing Systems, 36:31428–31449, 2023a.

Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 3836–3847, 2023b.

Qihui Zhang, Munan Ning, Zheyuan Liu, Yue Huang, Shuo Yang, Yanbo Wang, Jiayi Ye, Xiao Chen, Yibing Song, and Li Yuan. Upme: An unsupervised peer review framework for multimodal large language model evaluation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 9165–9174, 2025a.

Yinan Zhang, Eric Tzeng, Yilun Du, and Dmitry Kislyuk. Large-scale reinforcement learning for diffusion models. In European Conference on Computer Vision, pp. 1–17. Springer, 2024a.

Zechuan Zhang, Ji Xie, Yu Lu, Zongxin Yang, and Yi Yang. In-context edit: Enabling instructional image editing with in-context generation in large scale diffusion transformer. arXiv preprint arXiv:2504.20690, 2025b.

Zicheng Zhang, Haoning Wu, Erli Zhang, Guangtao Zhai, and Weisi Lin. Q-bench: A benchmark for multi-modal foundation models on low-level vision from single images to pairs. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2024b.

Haozhe Zhao, Xiaojian Shawn Ma, Liang Chen, Shuzheng Si, Rujie Wu, Kaikai An, Peiyu Yu, Minjia Zhang, Qing Li, and Baobao Chang. Ultraedit: Instruction-based fine-grained image editing at scale. Advances in Neural Information Processing Systems, 37:3058–3093, 2024.

Shitian Zhao, Qilong Wu, Xinyue Li, Bo Zhang, Ming Li, Qi Qin, Dongyang Liu, Kaipeng Zhang, Hongsheng Li, Yu Qiao, et al. Lex-art: Rethinking text generation via scalable high-quality data synthesis. arXiv preprint arXiv:2503.21749, 2025.

Kaiwen Zheng, Huayu Chen, Haotian Ye, Haoxiang Wang, Qinsheng Zhang, Kai Jiang, Hang Su, Stefano Ermon, Jun Zhu, and Ming-Yu Liu. Diffusionnft: Online diffusion reinforcement with forward process. arXiv preprint arXiv:2509.16117, 2025.

- A TRAINING SETTINGS We present key hyperparameters of the training in Table 5.

Parameter Setting

Basic Learning Rate 3e-4

- β1 0.9
- β2 0.999

Batch Size 3 EMA Decay 0.9

Sampling Sampling Inference Steps 6

Resolution 512 × 512 The Number of Images Per Prompt 12

The Number of Groups 24 DiffusionNFT

KL Loss Weight 0.0001 Guidance Strength (β1) 1.0

Group Filtering

τµ 0.9 τσ 0.05

Table 5: Key hyperparameters.

- B DETAILED HUMAN ALIGNMENT

We conducted a comprehensive human alignment study to rigorously evaluate the extent to which our proposed reward mechanism aligns with human judgment. We analyze this alignment from two perspectives: the accuracy of pairwise comparisons and the similarity of score distributions. For this part, we collected 800 edited images generated from 200 unique image-instruction pairs. These results were then annotated by three human evaluators across two distinct dimensions.

- B.1 REWARD METHOD DEFINITIONS

To comprehensively evaluate the human alignment of various reward mechanisms, we compared our proposed “Score Logit” method against several baselines. The details of each method are as follows:

Score Logit (Ours): This is our primary reward mechanism. We prompt the MLLM to evaluate the edit on a scale of 0 to 5. We then extract the logits corresponding to the score tokens (i.e., “0”, “1”, “2”, “3”, “4”, “5”), apply a softmax function to obtain a probability distribution, and compute the expected value (weighted average) as the raw reward score. This score is subsequently normalized to the range of [0, 1]. This process is detailed in Section 2.2.

Score Sampling (S D): In this approach, the MLLM is prompted to directly output a single numerical score from 1 to 5. This explicit score is then parsed and normalized to a value between 0 and 1 to serve as the reward.

Yes/No Logit (Y/N C): This method reframes the evaluation as a binary classification task. We prompt the MLLM to determine if the edit was successful (“Yes” or “No”). We then extract the logits for the “Yes” and “No” tokens, apply a softmax function, and use the resulting probability of “Yes” as the final reward signal.

Score Sampling + CoT: This is a variant of the discrete scoring method that incorporates Chainof-Thought (CoT) reasoning. The MLLM is first instructed to generate a brief analysis of the image

edit before outputting the final 0-5 score. The reward is derived from this final score, normalized to [0, 1].

Score Logit + CoT: Similar to the above, this method adds a CoT step to our continuous scoring approach. The MLLM first provides its reasoning, and then we extract the logits for the score tokens that follow the reasoning to calculate the weighted average reward.

Unified Reward (UR): As a strong baseline, we utilize the pre-trained reward model from the work of Wang et al. (2025b). We use the direct output of this model as the reward signal, without any modification or re-training.

- B.2 ALIGNMENT IN PAIRWISE PREFERENCE

First, we assess the alignment in terms of pairwise preference. For each input condition (original image and instruction), human annotators were asked to compare pairs of edited images and determine which one was better, or if they were of equivalent quality. This human judgment serves as the ground truth. We then measure the alignment of different reward models by calculating their pairwise accuracy—the percentage of pairs where the model’s preference (i.e., which image receives a higher reward) matches the human preference. A higher accuracy indicates that the reward model’s perception of relative quality is more similar to that of humans.

The results are presented in Figure 10. Our proposed method, “Score Logit”, which utilizes the expected value of score logits, achieves a pairwise accuracy of 74.74%. This result significantly surpasses all other baseline methods, including binary classification-based rewards and those using discrete scores. This demonstrates that our continuous reward signal is more effective at capturing the nuanced differences in edit quality that align with human perception.

We observe that incorporating Chain-of-Thought (CoT) unexpectedly degrades the performance of continuous scoring methods. We attribute it to a reasoning-induced bias, akin to the exposure bias in sequence generation (Pozzi et al., 2025). Recent studies suggest that when a model generates a lengthy reasoning chain, it can become overly persuaded by its own internally generated narrative, even if that narrative is flawed or drifts away from the initial evidence (Wu et al., 2023). In the context of image editing evaluation, the CoT process may cause the MLLM to focus more on its textual justification rather than the subtle but critical visual details of the images. Consequently, its final judgment becomes less grounded. We enforce a direct evaluation with a restricted output format, which effectively mitigates this bias and ensures the assessment remains tightly coupled with the visual input.

Background

Color

Material Alter

Motion

Style

Add

Remove

Replace

25%

50%

75%

S_L

UR

S_L+CoT S_S+CoT Y/N_L

S_S

Figure 10: Performance comparison across different editing tasks.

Background

Color

Material Alter

Motion

Style

Add

Remove

Replace

25%

50%

75%

Qwen2.5VL-32b

InternVL-14b

Qwen2.5VL-7b Qwen2.5VL-3b

Figure 11: Performance comparison across different models.

- B.3 ALIGNMENT IN SCORE DISTRIBUTION

We analyze the alignment of score distributions. In this task, annotators assigned a quality label (“Good”, “Bad”, or “Indistinguishable”) and an absolute quality score from 1 to 5 to each edited

image. We evaluate alignment from two aspects. First, we compare the overall distribution of scores generated by each reward model against the distribution of scores given by human annotators. As shown in Figure 12, a high degree of similarity in distribution suggests that the reward model shares a similar tendency and preference scale with humans.

The results clearly indicate the superiority of “Score Logit”. The figures show that its score distribution most closely mirrors that of human evaluators. Furthermore, it exhibits strong consistency with human judgment, assigning significantly higher scores to “Good” edits and lower scores to “Bad” ones compared to the other methods. Both analyses confirm that “Score Logit” provides a reward signal that is more accurate in relative comparisons and better calibrated to the absolute scale of human quality perception, leading to a higher degree of alignment.

[Figure 100]

0.7

0.13 0.04 0.05 0.06 0.73

Yes/No Logit

0.6

0.06 0.01 0.14 0.25 0.54

Score Sampling

0.5

0.08 0.08 0.05 0.07 0.72

Score Logit + CoT

0.4

0.00 0.05 0.33 0.60 0.02

UnifiedReward

0.3

0.06 0.09 0.06 0.17 0.63

Score Sampling + CoT

0.2

- 0.04 0.04 0.09 0.24 0.60
- 0.05 0.05 0.09 0.22 0.59

Score Logit

0.1

human

0.00 0.25 0.50 0.75 1.00

Figure 12: Score distribution across different reward methods

- B.4 ANNOTATION DETAILS

To construct a robust ground truth for evaluating the human alignment of our reward mechanism, we undertook a detailed manual annotation process. We collected 800 edited images generated from 200 unique image-instruction pairs for this study. The annotation was carried out by a team of five human evaluators, all of whom are proficient in English and possess a strong understanding of the image editing domain.

To ensure the reliability and consistency of our annotations, we implemented a rigorous quality control protocol. Each annotation task was initially assigned to three distinct evaluators to ensure multiple perspectives. A label was accepted if a consensus was reached, requiring at least two of the three annotators to provide the same judgment. In cases where no initial consensus was reached, the task was reassigned to two of the remaining evaluators who had not previously assessed the sample. A final label was considered valid only if an overall agreement rate of 60% or higher was achieved across all assigned evaluators. Samples that failed to meet this threshold were discarded from our analysis to maintain the high quality and integrity of the ground-truth data.

The evaluators performed two primary annotation tasks corresponding to the dimensions of our human alignment analysis. For pairwise preference comparison, annotators were presented with a pair of edited images for a given original image and editing instruction. They were tasked with selecting which image better fulfilled the instruction or labeling them as being of equivalent quality. This data forms the basis for our pairwise accuracy evaluation. For absolute quality scoring, annotators evaluated individual edited images, assigning both a categorical label (‘Good’, ‘Bad’, or ‘Indistinguishable’) and an absolute quality score on a scale of 1 to 5. This data was used to analyze the alignment of score distributions between reward methods and human judgment.

- C PROMPT TEMPLATE Template for Yes/No Evaluation

System Prompt: You are a helpful assistant. User Prompt: Here are two images: the original and the edited version. Please evaluate if the edited image successfully meets the following editing instructions and requirements. Instruction: {{instruction}} Requirements: {{requirement}}

You need to provide a “Yes” or “No” judgment based on the accuracy and quality of the edit. Answer “Yes” if: The correct object was edited according to the instruction, all requirements were met, and the visual result is high quality. Answer “No” if: The wrong object was edited, the edit fails to meet the requirements, or the visual quality is poor.

Response Format: “Judgement”: [[Yes/No]]

### Template for Score Evaluation

System Prompt: You are a helpful assistant. User Prompt: Here are two images: the original and the edited version. Please evaluate the edited image based on the following editing instructions and requirements. Instruction: {{instruction}} Requirements: {{requirement}}

You need to rate the editing result from 0 to 5 based on the accuracy and quality of the edit. 0: The wrong object was edited, or the edit completely fails to meet the requirements. 5: The correct object was edited, the requirements were met, and the visual result is high quality.

Response Format: “Score”: [[rating]](0-5).

### Template for CoT Score Evaluation

System Prompt: You are a helpful assistant. User Prompt: Here are two images: the original and the edited version. Please evaluate the edited image based on the following editing instruction and requirements. Instruction: {{instruction}} Requirements: {{requirement}}

You need to rate the editing result from 0 to 5 based on the accuracy and quality of the edit. Before that, analyze the image and provide some reasoning process for better evaluation (keep your reasoning concise). 0: The wrong object was edited, or the edit completely fails to meet the requirements. 5: The correct object was edited, the requirements were met, and the visual result is high quality.

Response Format: “Reasoning”: “...”, “Score”: [[rating]](0-5).

