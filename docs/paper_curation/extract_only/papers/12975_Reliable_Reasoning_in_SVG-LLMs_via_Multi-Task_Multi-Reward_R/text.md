## arXiv:2603.16189v1[cs.CV]17Mar2026

[Figure 1]

# Reliable Reasoning in SVG-LLMs via Multi-Task Multi-Reward Reinforcement Learning

##### Haomin Wang1,2*, Qi Wei2,3*, Qianli Ma1,2, Shengyuan Ding2,4, Jinhui Yin2,3, Kai Chen2 and Hongjie Zhang2†

1Shanghai Jiao Tong University, 2Shanghai AI Laboratory, 3Nanjing University, 4Fudan University

#### Abstract

With the rapid advancement of vision–language models, an increasing number of studies have explored their potential for SVG generation tasks. Although existing approaches improve performance by constructing large-scale SVG datasets and introducing SVG-specific tokens, they still suffer from limited generalization, redundant paths in code outputs, and a lack of explicit reasoning. In this work, we present CTRL-S (Chain-of-Thought Reinforcement Learning for SVG), a unified framework that introduces a chain-of-thought mechanism to explicitly expose the model’s reasoning process during SVG generation. To support this structured reasoning, we construct SVG-Sophia, a high-quality dataset containing 145K samples across SVG code refinement, Text-to-SVG, and Image-to-SVG tasks. By training the model to generate group-level structured SVG code, CTRL-S significantly improves structural coherence and visual fidelity. Furthermore, we adopt the GRPO algorithm and design a multi-reward optimization framework, incorporating DINO, image–text similarity, format, and code efficiency rewards. Through joint multi-reward optimization and multi-task training, our approach systematically enhances overall generation capabilities. Extensive experiments show that CTRL-S outperforms existing methods, achieving higher task success rates, superior SVG code quality, and exceptional visual fidelity.

GitHub Repo https://github.com/hmwang2002/CTRL-S

SVG-Sophia Dataset https://huggingface.co/datasets/InternSVG/SVG-Sophia

### 1. Introduction

Scalable Vector Graphics (SVG) is an XML-based vector format that represents 2D content using parameterized geometric primitives rather than pixel grids, offering compact storage, resolution independence, and fine-grained editability. Owing to its seamless integration with modern front-end systems and interactive frameworks, SVG has become a fundamental graphic medium in web design, user interface development, scientific visualization, and computer-aided design.

With the rapid development of vision-language models [2, 10, 16, 17, 34, 44], recent research has begun to explore their application to high-quality SVG code generation [24, 33, 37, 41]. By integrating vision encoders and SVG-specific tokens, these approaches significantly improve performance on Text-to-SVG and Image-to-SVG tasks. However, these approaches still suffer from limited generalization, frequently producing SVG programs with redundant paths. In addition, overly aggressive code compression during training degrades the readability and editability of the generated vector graphics. SVGen [32] and SVGThinker [4] introduce the chain-of-thought (CoT) reasoning into SVG generation by explicitly exposing intermediate reasoning steps to improve the quality of the generated SVG. However, they do not fully exploit the inherent grouping (<g>) structures in SVG code to organize components hierarchically, nor do they establish a clear alignment between reasoning steps and the corresponding grouped code segments, resulting in limited structural transparency and editability. While

* Equal Contribution: kiyotakawang@sjtu.edu.cn, qiwei@smail.nju.edu.cn † Corresponding author: nju.zhanghongjie@gmail.com

###### Multi-Task Multi-Reward GRPO Training

Generated Examples

DINO Reward

Text-to-SVG

Image-text Reward

Multi-Task

###### Multi-Reward

Image-to-SVG

Format Reward

Code Refinement

Code eff Reward

CTRL-S

<svg xmlns="http://www.w3.org/2000/svg" width="128" height="128" viewBox="0 0 128 128">

<think> Canvas: 128 x 128 Image Description: A minimalist breakfast or snack scene featuring ...... Drawing Steps:

You are an expert in SVG generation. You need to generate an SVG with the following format: <think>

Refinement Examples

- 1. Yellow oval tray - The base platform/tray with a 3D effect created by a darker bottom edge
- 2. Teal leaf - The apple's leaf element positioned at the top

<!-- Yellow Oval Tray - Base platform holding all items -->

<g id="tray">

<path fill="#ffee93" d="..."/> </g>

<!-- Teal Leaf - Apple's decorative leaf element --> <g id="apple-leaf">

<path fill="#29d0b3" d="..."/> </g>

- 3. Red apple - The apple shape on the right side, overlapping the

- - Canvas dimensions
- - Image description
- - Step-by-step breakdown

</think> ```svg [Modular SVG with clear comments] ``` Let's think step by step and generate the SVG based on the instruction below. Instruction: A minimalist breakfast or snack scene featuring a light blue mug, a peachcolored bread loaf, and a red apple with a teal leaf, all arranged on a yellow oval tray.

tray

<!-- Red Apple - Right side fruit element --> <g id="apple">

- 4. Peach bread loaf - The rounded bread shape in the center
- 5. Light blue mug (back portion) - The left side/back part of the mug handle area
- 6. Light blue mug (main body) - The main cylindrical body of the mug </think>

<path fill="#ea4e4e" d="..."/> </g>

... </svg>

###### Chain-of-Thought Reasoning Group-level Structured SVG Code

Step 1 Step 2 Step 3 Step 4 Step 5 Step 6

Detailed Instruction

- Figure 1: Overview of CTRL-S. (Top Left) The Multi-Task Multi-Reward GRPO training framework integrates diverse generation tasks (Text-to-SVG, Image-to-SVG, and Code Refinement) guided by multiple rewards. (Bottom Left) During inference, CTRL-S leverages chain-of-thought reasoning to plan step-by-step drawing operations before generating the final group-level structured SVG code, ensuring a clear one-to-one correspondence between the reasoning steps and the generated code groups. (Right) Examples of high-quality generated SVGs and successful code refinement processes.

recent works like RLRF [25] and Reason-SVG [36] incorporate the GRPO algorithm [27] to leverage visual reward signals during post-training reinforcement learning, they primarily optimize individual tasks in isolation and lack a unified framework for jointly training Text-to-SVG and Image-to-SVG generation.

To address these limitations, we propose CTRL-S, a unified framework tailored for Text-to-SVG, Imageto-SVG, and SVG code refinement tasks. As illustrated in Figure 1, we integrate CoT reasoning into SVG generation to expose the model’s planning processes. By leveraging the inherent grouping characteristics of SVG, we establish a step-wise alignment between the reasoning steps and the corresponding code groups. Furthermore, to break the isolation of prior works that exclusively focus on single-task optimization, we not only jointly train the Text-to-SVG and Image-to-SVG tasks but also introduce an SVG code refinement task. By endowing the model with self-diagnostic and error-correction capabilities, these three tasks mutually reinforce each other within a single unified model.

To facilitate this unified paradigm, we first construct SVG-Sophia, a high-quality dataset that encompasses CoT question-answering pairs across the three tasks. Comprising 131K SFT samples and 14.4K RL samples, SVG-Sophia provides a solid foundation for CTRL-S to excel in these diverse domains. In the RL post-training phase, we address the limitations of conventional SFT, which relies solely on token-level supervision and lacks visual feedback. We introduce a multi-task, multi-reward optimization framework based on the GRPO algorithm. Specifically, we design four complementary rewards: (1) a format reward to ensure structural validity and renderability, (2) a DINO reward to encourage deep visual feature alignment between the rendered SVG and the reference image, (3) an image–text similarity reward to promote semantic consistency between the generated SVG and the input instruction, and (4) a code efficiency reward to penalize unnecessarily verbose SVG outputs and improve inference efficiency. This multi-reward optimization not only enhances visual fidelity but also mitigates the repetitive code generation commonly observed in prior SVG-LLM models, achieving a balanced trade-off between reasoning efficiency and generation quality. Extensive experiments show that our multi-task, multi-reward RL algorithm yields significant gains over SFT. Joint multi-task training further improves performance and generalization compared to single-task optimization. Moreover, the introduction of CoT enhances generation success and visual quality for complex geometries, while transforming the implicit generation process into explicit, structured code blocks, substantially improving the readability and editability of the resulting SVGs. In summary, our contributions are as follows:

- 1. We propose CTRL-S, a unified framework that integrates chain-of-thought reasoning and multi-task, multi-reward online RL for SVG code refinement, Text-to-SVG, and Image-to-SVG tasks.

- 2. We construct SVG-Sophia, a high-quality dataset providing explicit chain-of-thought supervision across three SVG tasks.
- 3. Extensive experiments show that our multi-task, multi-reward RL framework achieves substantial performance gains over SFT baselines. CTRL-S achieves state-of-the-art performance in SVG generation, delivering higher visual quality, faster inference, and highly readable and editable code.

### 2. Related Work

Optimization-based SVG Modeling. Optimization-based methods formulate SVG modeling as a parameter optimization problem rather than training a dedicated generative model. Early works such as Diff VG [13] and LIVE [15] leverage differentiable rasterization to directly optimize Bézier control points and styling attributes by minimizing pixel-level reconstruction losses. To incorporate semantic supervision, CLIP-based approaches [7, 26, 29–31] replace pixel losses with image-text similarity objectives, enabling text-conditioned SVG generation without training. More recently, Score Distillation Sampling (SDS) [20] has been adopted to transfer diffusion priors into the vector graphics domain [11, 38–40, 43]. These methods optimize rendered SVGs through gradients derived from pretrained diffusion models, with later variants such as VPSD introducing particle-based distributional optimization to improve diversity and stability. Despite their strong visual fidelity, optimizationbased approaches remain computationally intensive, instance-specific, and lack explicit hierarchical modeling of SVG structure, limiting scalability and downstream editability.

Learning-based SVG Modeling. Early learning-based methods represent SVG as sequences of geometric primitives and adopt task-specific generative architectures [3, 9, 14, 22, 23, 28]. Sketch-RNN [9] models drawings as sequential pen trajectories, SVG-VAE [14] introduces latent-variable modeling for vector synthesis, and DeepSVG [3] employs hierarchical VAEs with Transformer decoders to capture global layouts and path-level details. With the emergence of large language models (LLMs) and vision-language models (VLMs), recent research has shifted toward semantically grounded SVG generation [4–6, 12, 24, 25, 32, 33, 35–37, 41, 45]. Methods like StarVector [24], LLM4SVG [37], OmniSVG [41], and InternSVG [33] incorporate vision encoders and SVG-specific tokens to support Text-to-SVG and Image-to-SVG generation. Moreover, recent works such as SVGen [32] and SVGThinker [4] aim to introduce chain-of-thought reasoning into SVG generation by explicitly exposing intermediate reasoning steps, thereby improving performance. However, they fail to fully exploit the inherent grouping characteristics of SVG code to establish a one-to-one alignment between the intermediate planning steps and the generated code blocks.

Reinforcement Learning for SVG Generation. Beyond standard supervised fine-tuning, applying reinforcement learning (RL) during the post-training stage has emerged as a promising frontier for SVG generation. Recent works such as RLRF [25] and Reason-SVG [36] adopt the GRPO algorithm [27], introducing visual reward signals to further enhance generative quality. However, these approaches remain confined to single-task optimization, failing to unify Text-to-SVG and Image-to-SVG generation under a shared paradigm. In contrast, our CTRL-S introduces a unified, multi-task RL optimization framework that jointly aligns Text-to-SVG, Imageto-SVG, and SVG code refinement within a single unified model.

### 3. SVG-Sophia

We collect the original SVG files from the ColorSVG-100K [6] dataset and leverage Claude-Sonnet-4.5 [1] to annotate them into high-quality samples with explicit chain-of-thought reasoning and group-level structured SVG code. For Text-to-SVG generation, we construct 50K SFT samples and 5.5K RL samples. For Image-to-SVG generation, we similarly build 50K SFT samples and 5.5K RL samples, sharing the same underlying SVG programs as Text-to-SVG but differing in input modality. For SVG code refinement, we curate 31K SFT samples and 3.4K RL samples, along with a test set of 934 samples.

##### 3.1. Task Definition

Let ℳ denote the MLLM and 𝐼text represent the user-provided textual instruction. For the Text-to-SVG generation task, the model is tasked with autoregressively generating a CoT planning sequence 𝑂think, followed

by the corresponding executable SVG code 𝑂svg. This process is defined as:

(𝑂think,𝑂svg) = ℳ(𝐼text) (1)

Similarly, for the Image-to-SVG generation task, the model is additionally conditioned on a reference image 𝐼image. The task is formulated as:

(𝑂think,𝑂svg) = ℳ(𝐼text,𝐼image) (2)

To empower the model with self-correction and optimization capabilities, we introduce the SVG code refinement task. In this setting, the model is provided with a textual instruction 𝐼text, a reference image 𝐼image, and a flawed SVG code draft 𝐼draft to be refined:

(𝑂think,𝑂refined) = ℳ(𝐼text,𝐼image,𝐼draft) (3)

##### 3.2. Data Annotation Pipeline

The raw SVG files are initially collected from the ColorSVG-100K [6] dataset and then normalized to a 128×128 viewBox. We employ Claude-Sonnet-4.5 [1] to annotate detailed image captions from the rendered vector graphics. Subsequently, we prompt Claude-Sonnet-4.5 with both the generated caption and the raw SVG code, instructing it to refactor the original code into a highly structured format, enriched with descriptive comments and semantic group-level hierarchies, while also producing a step-by-step reasoning process that outlines its planning procedure. To ensure strict visual fidelity and eliminate failed refactoring attempts, we filter the refactored SVGs by retaining only those achieving an SSIM ≥ 0.95 against their original renderings. To further ensure annotation quality, we engage 100 human annotators to review all annotated samples, manually correcting any captions that inaccurately describe the visual content or CoT reasoning steps that fail to correspond to the generated code groups. Finally, we use the generated image captions as user instructions and treat the CoT reasoning along with the reconstructed structured SVG code produced by Claude-Sonnet-4.5 as the ground-truth responses for Text-to-SVG and Image-to-SVG tasks.

For the SVG code refinement task, we first train a Qwen3-VL-8B model [2] on the annotated Text-to-SVG and Image-to-SVG data, and use it to generate draft SVG programs on the training set. We then retain only moderately flawed samples (0.30 ≤ SSIM ≤ 0.95) against the ground truth. Claude-Sonnet-4.5 is then prompted with the defective and ground-truth images to produce discrepancy analysis and correctionoriented CoT reasoning. Rule-based filtering is further applied to remove invalid annotations, such as cases claiming complete consistency or providing irrelevant analysis. To mitigate potential annotator bias, 100 human annotators further review all refinement annotations, manually correcting cases where the identified defects or correction reasoning are inaccurate or task-irrelevant. For the test set, we select non-overlapping SVG programs from ColorSVG-100K and apply the same annotation pipeline. Defective drafts are generated using the SFT-trained Qwen3-VL-8B, as well as Claude-Sonnet-4.5, Gemini-3-Pro [8], GPT-5.2 [18], and Qwen3-VL-235B-A22B [2], to ensure a fair and unbiased evaluation.

### 4. CTRL-S

- Figure 2 illustrates the overall pipeline of CTRL-S. Our framework begins with a two-stage supervised fine-tuning to align SVG-specific tokens and establish step-wise chain-of-thought reasoning. Subsequently, a multi-task, multi-reward reinforcement learning phase jointly optimizes Text-to-SVG, Image-to-SVG, and code refinement tasks via comprehensive feedback signals.

##### 4.1. Preliminary

Notation and Problem Formulation. We formulate SVG generation as a unified multi-task sequence-tosequence autoregressive generation problem. Let ℳ (defined in Sec. 3.1) parameterized by 𝜃 denote our MLLM. Depending on the specific task, the model is conditioned on a varying set of inputs to generate a target sequence 𝑦 = (𝑦1,𝑦2,...,𝑦𝑇), which consists of a chain-of-thought reasoning sequence 𝑂think followed by the

###### I. Two-Stage Supervised Fine-Tuning

Stage 1: SVG Token Alignment Stage 2: CoT-Structured Alignment

Step-wise Alignment

<think>

- 1. object 1
- 2. object 2
- 3. object 3 </think>

- <g id="object 1>...
- <g id="object 2>...
- <g id="object 3>...

Train SVG-specific Token Embeddings

SAgoge (1M)

CoT-Enabled SVG-LLM

SVG-specific Tokens

Token Aligned SVG-LLM

Train CoT Responses format

SVG-Sophia (131K)

###### II. Multi-Task Multi-Reward Reinforcement Learning

Image-text Similarity Reward

Rollout Rendered Format Reward

Long-CLIP Image Encoder

CoT+SVG

Image

Text-to-SVG

Exact one <think> & one <svg>?

Rlclip

Generated

Instruction

CoT+SVG

Image

Renderable?

Long-CLIP Text Encoder

Image-to-SVG

Instruction

CoT+SVG

Image

Rformat:'1' or '0'

Image

###### ...

...

###### DINO Reward

###### Code Eff Reward

Lgen

Refinement

###### Policy Model

CoT+SVG

Image

Generated DINOv2 Rdino

Reff

Flawed SVG

Lgt

GT

###### Reference Model Advantage Group Computation Rtotal=Rformat·(wdinoRdino+wlclipRlclip+weffReff)

KL Divergence

- Figure 2: The overall pipeline of CTRL-S. (1) Two-Stage SFT: The model is first trained on 1M SAgoge samples to align SVG-specific tokens, and then fine-tuned on SVG-Sophia to learn CoT-structured responses with explicit step-wise planning. (2) Multi-Task Multi-Reward RL: We jointly optimize Text-to-SVG, Image-to-SVG, and SVG refinement tasks via a multi-reward mechanism, including Format Reward, DINO Reward, Imagetext Similarity Reward, and Code Efficiency Reward, to improve structural validity, visual fidelity, semantic alignment, and concise code generation.

executable SVG code (𝑂svg or 𝑂refined). To unify our three core tasks, 𝑐 encapsulates varying inputs: 𝑐 = 𝐼text for Text-to-SVG, 𝑐 = (𝐼text,𝐼image) for Image-to-SVG, and 𝑐 = (𝐼text,𝐼image,𝐼draft) for SVG Code Refinement. Given the task-specific context 𝑐, the generation probability of the output sequence 𝑦 is factorized as:

∏︁𝑇

𝜋𝜃(𝑦𝑡|𝑐,𝑦<𝑡), (4)

𝑃(𝑦|𝑐) =

𝑡=1

where 𝑦<𝑡 represents the sequence of tokens generated prior to step 𝑡. The model, typically initialized after multi-task SFT, serves as our reference policy 𝜋𝑟𝑒𝑓 for the reinforcement learning phase.

Group Relative Policy Optimization (GRPO). To efficiently optimize the MLLM across diverse tasks without the memory overhead of a parameterized value model, we employ GRPO [27]. For a given context 𝑐, the current policy 𝜋𝜃

samples a group of 𝐺 diverse output trajectories 𝒢 = {𝑦(1),...,𝑦(𝐺)}. Each trajectory 𝑦(𝑖) is evaluated by our multi-reward function to yield a score 𝑟𝑖. GRPO computes the relative advantage by normalizing these rewards within the group: 𝐴ˆ𝑖 = (𝑟𝑖 − 𝜇𝒢)/(𝜎𝒢 + 𝜖). The policy 𝜋𝜃 is then optimized by maximizing a clipped surrogate objective, augmented with a Kullback-Leibler (KL) divergence penalty to mitigate excessive deviation from 𝜋𝑟𝑒𝑓:

𝑜𝑙𝑑

⎡ ⎣ 1

⎤ ⎦, (5)

|∑︁𝑦(𝑖)|

∑︁𝐺

1 |𝑦(𝑖)|

ℒ𝑖,𝑡𝑐𝑙𝑖𝑝(𝜃) − 𝛽D𝐾𝐿(𝜋𝜃‖𝜋𝑟𝑒𝑓)

###### 𝒥𝐺𝑅𝑃𝑂(𝜃) = E𝑐∼𝒟,𝒢∼𝜋

𝜃𝑜𝑙𝑑

𝐺

𝑡=1

𝑖=1

where the clipped likelihood ratio is defined as

ℒ𝑖,𝑡𝑐𝑙𝑖𝑝(𝜃) = min(︁𝜌𝑖,𝑡(𝜃)𝐴ˆ𝑖,clip(𝜌𝑖,𝑡(𝜃),1 − 𝛾,1 + 𝛾)𝐴ˆ𝑖)︁, (6)

(𝑖) 𝑡 |𝑐,𝑦<𝑡(𝑖))

and 𝜌𝑖,𝑡(𝜃) = 𝜋𝜃(𝑦

is the probability ratio of generating the 𝑡-th token under the current versus the old policy.

𝜋𝜃𝑜𝑙𝑑(𝑦𝑡(𝑖)|𝑐,𝑦<𝑡(𝑖))

##### 4.2. Two-Stage Supervised Fine-Tuning

To establish a robust initialization for the subsequent reinforcement learning phase, CTRL-S adopts the SVGspecific token design introduced in InternSVG [33] (detailed in the Appendix) and undergoes a two-stage SFT process. In the first stage, we stabilize the embeddings of the SVG-specific tokens by sampling 1M training instances from the SAgoge dataset [33]. Following this modality alignment, the second stage utilizes the SFT split of the SVG-Sophia dataset to train the model. This phase introduces a strict step-wise alignment, where each intermediate reasoning step in the CoT explicitly corresponds to a hierarchically organized, group-level (<g>) structural block in the resulting SVG, ensuring that the SVG generation process is both interpretable and logically transparent.

##### 4.3. Multi-Reward Design for Reinforcement Learning in CTRL-S

Following the SFT phase, we employ reinforcement learning to further align the model’s generation with visual, semantic, and structural objectives. To provide comprehensive guidance without relying on costly human annotations, we design a multi-reward framework comprising four complementary components.

Format Reward (𝑅format) To guarantee both structural compliance and execution validity, we introduce a binary format reward 𝑅format. The reward yields 1 if the model’s output strictly contains exactly one <think>...</think> reasoning block followed by a single SVG code block that can be rendered by CairoSVG successfully, and 0 otherwise.

DINO Reward (𝑅dino) A primary limitation of standard SFT is its inherent reliance on token-level textual supervision, which lacks the capacity to penalize global visual discrepancies. For SVG-related tasks, explicit pixel-level feedback is crucial to enhance the overall visual fidelity of the generated graphics. To address this, we introduce 𝑅dino. Specifically, the generated SVG code is first rasterized into an image 𝑉gen. We then compute the feature similarity between this rendering and the ground-truth image 𝑉gt using a pre-trained DINOv2 [19] model, capturing deep, structural visual alignments. Formally, let ℰDINO denote the DINOv2 feature extractor; the reward is formulated as the normalized cosine similarity between the two image embeddings:

- 1

- 2

(cos(ℰDINO(𝑉gen),ℰDINO(𝑉gt)) + 1). (7)

𝑅dino =

Image-text Similarity Reward (𝑅lclip) Beyond low-level visual fidelity (Eq. 7), the generated SVG must semantically align with the user’s high-level textual instruction 𝐼text. Considering that the instructions in SVG-Sophia typically consist of several detailed descriptive sentences, the standard CLIP model [21], bounded by its strict 77-token input limit, often truncates crucial structural details and fails to adequately capture fine-grained semantics in long contexts. To overcome this, we adopt Long-CLIP [42] to compute the semantic alignment reward 𝑅lclip. By leveraging the Long-CLIP image encoder ℰL-img and text encoder ℰL-text, we project both the rendered image 𝑉gen and the instruction 𝐼text into a shared embedding space. The reward is computed as follows:

𝑅lclip = ℰL-img(𝑉gen) ‖ℰL-img(𝑉gen)‖2

ℰL-text(𝐼text) ‖ℰL-text(𝐼text)‖2

. (8)

·

Code Efficiency Reward (𝑅eff) During the generation of SVG code, SFT models frequently suffer from a repetition problem, producing excessively long, redundant, and invalid code that significantly degrades inference speed. To mitigate this issue, we adapt a length-based penalty inspired by RLRF [25]. Specifically, let 𝐿gt and 𝐿gen denote ground-truth and generated SVG code lengths, the code efficiency reward 𝑅eff is formulated as follows:

1 𝐿gt

𝑅eff = 1 − (

max(0,𝐿gen −

𝐿gt 2

))2. (9)

Total Reward (𝑅total) Finally, we aggregate the visual (Eq. 7), semantic (Eq. 8), and efficiency objectives (Eq. 9) into a unified multi-reward formulation. Crucially, the binary format reward 𝑅format acts as a multiplicative

Table 1: SVG generation results on SArena-Icon. SR denotes Success Rate. CTRL-S (SFT) denotes the model after two-stage SFT, CTRL-S (SFT+RL) denotes the full model after RL post-training.

Text-to-SVG Image-to-SVG

Model Data

FID ↓ CLIP-T2I ↑ CLIP-I2I ↑ SR Tokens DINO ↑ SSIM ↑ LPIPS ↓ SR Tokens Traditional SVG Methods

IconShop – 32.288 20.894 70.922 86.51% 1.6k – – – – – VectorFusion – 16.594 22.992 70.308 100.0% 33k – – – – – SVGDreamer – 26.612 20.329 69.975 100.0% 132k – – – – –

Diff VG – – – – – – 0.869 0.927 0.097 100.0% 17k LIVE – – – – – – 0.973 0.986 0.024 100.0% 18k VTracer – – – – – – 0.966 0.875 0.054 100.0% 4.4k

###### Vision-Language Models

Llama 4 Scout – 17.908 22.849 73.563 96.03% 256 0.844 0.582 0.346 97.61% 246 Llama 4 Maverick – 14.931 23.570 75.816 98.12% 265 0.863 0.596 0.329 97.75% 255

Qwen3-VL-235B-A22B – 18.404 24.024 76.289 94.96% 326 0.927 0.718 0.210 95.28% 679

GPT-5.2 – 14.965 25.457 82.701 97.02% 623 0.936 0.653 0.269 98.87% 497 Claude-Sonnet-4.5 – 16.451 25.322 81.038 99.63% 320 0.910 0.677 0.253 97.95% 705

Gemini-3-Pro – 13.203 25.731 84.157 96.91% 397 0.940 0.723 0.198 97.14% 769

###### SVG-LLMs

StarVector-8B 2.2M – – – – – 0.871 0.623 0.206 72.51% 951 LLM4SVG-7B 580K 21.939 19.458 70.726 90.95% 705 0.748 0.472 0.409 95.53% 485 OmniSVG-3B 2M 28.292 21.679 74.831 99.68% 1.8k 0.894 0.756 0.186 99.97% 2.4k InternSVG-8B 16M 8.715 23.916 80.911 97.24% 1.0k 0.949 0.811 0.127 94.45% 1.3k

###### Ours

Qwen3-VL-8B – 16.875 22.578 73.187 99.55% 374 0.796 0.500 0.354 74.02% 474 CTRL-S (SFT) 131K 18.154 23.655 77.959 92.02% 1.1k 0.903 0.717 0.187 90.12% 1.4k

###### CTRL-S (SFT+RL) 145K 11.584 25.944 82.291 99.85% 346 0.980 0.835 0.098 99.93% 512

Δ Improvement 6.570 2.289 4.332 0.077 0.118 0.089

gating factor, ensuring that unrenderable or structurally malformed outputs receive a total reward of zero, preventing degenerate policy updates. The final reward is defined as:

𝑅total = 𝑅format · (︀

)︀

. (10) Empirically, we set the trade-off weights as 𝑤dino : 𝑤lclip : 𝑤eff = 2 : 1 : 1.

𝑤dino𝑅dino + 𝑤lclip𝑅lclip + 𝑤eff𝑅eff

### 5. Experiments

##### 5.1. Experimental Setup

Building upon Qwen3-VL-8B-Instruct, CTRL-S initially undergoes a two-stage SFT process, as detailed in Sec. 4.2. We set the learning rate to 1e-4 in the first stage and decrease it to 5e-5 in the second stage. The training is performed on 48 H200 GPUs with a global batch size of 96. In the RL stage, we optimize the model using the GRPO algorithm implemented via the verl framework. The RL training is performed on 32 GPUs with a global batch size of 128 and a learning rate of 1e-5. During the rollout phase, we sample 16 responses per prompt. The model is trained for 2 epochs, and the entire RL training process takes approximately 12 hours.

##### 5.2. Quantitative Evaluations

As shown in Table 1, CTRL-S achieves leading performance on both the Text-to-SVG and Image-to-SVG tasks on the SArena-Icon [33] benchmark. For Text-to-SVG, CTRL-S attains the highest CLIP-T2I score of 25.944, demonstrating the outstanding semantic understanding and text-to-image alignment capabilities of the model.

Table 2: SVG refinement results on SVG-Sophia Code Refinement Benchmark.

Model DINO ↑ SSIM ↑ LPIPS ↓ SR Tokens Vision-Language Models

Llama 4 Scout 0.840 0.634 0.383 97.00% 788 Llama 4 Maverick 0.845 0.615 0.377 94.75% 616

Qwen3-VL-235B-A22B 0.809 0.515 0.403 77.84% 840

GPT-5.2 0.911 0.640 0.342 99.26% 975 Claude-Sonnet-4.5 0.796 0.579 0.401 84.58% 790

Gemini-3-Pro 0.883 0.593 0.284 81.37% 992

###### Ours

Qwen3-VL-8B 0.796 0.501 0.410 76.77% 980 CTRL-S (SFT) 0.888 0.665 0.236 84.37% 2.9k

###### CTRL-S (SFT+RL) 0.951 0.765 0.180 99.79% 866

Δ Improvement 0.067 0.126 0.104

For Image-to-SVG, our model obtains the best results across the DINO, SSIM, and LPIPS metrics compared to mainstream general VLMs and SVG-LLMs. This proves that CTRL-S exhibits extremely high visual fidelity in accurately reconstructing geometric shapes and colors. Furthermore, compared to our SFT baseline, CTRL-S (SFT + RL) not only gains significant enhancements across all visual metrics but also substantially increases the success rate and greatly reduces the number of generated tokens. This validates the effectiveness of our proposed multi-task, multi-reward RL training framework. The framework successfully enhances visual fidelity and semantic alignment while simultaneously empowering the model with stronger generalization capabilities, robustness, and superior code generation efficiency.

As shown in Table 2, on the SVG-Sophia Code Refinement Benchmark, CTRL-S achieves the best performance across all evaluation metrics compared with state-of-the-art proprietary models, including GPT-5.2, ClaudeSonnet-4.5, and Gemini-3-Pro. These results highlight the superiority of CTRL-S in structural understanding and fine-grained code editing. Specifically, starting from imperfect SVG programs, our method accurately locates defective components and applies targeted corrections, achieving more precise structural refinement while preserving semantic consistency. This indicates that the model not only possesses a strong understanding of the visual information encoded in SVG programs but also demonstrates stable, iterative code-optimization capability. Furthermore, compared with the SFT baseline, the RL-enhanced model achieves a substantial improvement in task success rate while significantly reducing the number of generated tokens. This suggests better generalization and a reduced tendency to produce redundant code, leading to more efficient and controllable refinement.

##### 5.3. Qualitative Evaluations

As shown in Figure 3, we present qualitative comparisons between CTRL-S and representative baselines across three tasks. For Text-to-SVG and Image-to-SVG tasks (Figures 3a and 3b), CTRL-S consistently generates SVG with accurate structural layouts and fine-grained visual details. For Text-to-SVG, CTRL-S accurately renders complex multi-element scenes, such as a striped hot air balloon with surrounding clouds and a suspended basket. In contrast, competing methods frequently distort the global structure, omit essential components, or fail to reproduce the correct color patterns. For Image-to-SVG, CTRL-S faithfully reconstructs reference images while preserving color attributes and spatial arrangements. In contrast, general VLMs often produce geometries that are structurally inconsistent and physically implausible, while SVG-LLMs such as InternSVG-8B exhibit noticeable deviations in stroke thickness and shape boundaries. For SVG code refinement (Figure 3c), CTRL-S demonstrates strong self-correction capability by accurately identifying structural defects in the draft SVG and applying targeted modifications. In contrast, general VLMs often fail to precisely localize defective components, resulting in incomplete or globally inconsistent corrections. Starting from imperfect drafts, CTRL-S successfully recovers missing components and corrects spatial misalignments, producing refined outputs that closely match the ground truth. In Figure 4, we further illustrate the progressive improvement of CTRL-S throughout RL training. As the number of training steps increases, the generated SVGs exhibit increasingly accurate structural

Claude-Sonnet-4.5Gemini-3-Pro GPT-5.2 Qwen3-VL-235B-A22BInternSVG-8B CTRL-S(Ours)

A single open envelope in front view with a white letter emerging from the top, displaying four horizontal coral lines representing text. Flat fills in light pink, white, and coral create a clean, minimalist aesthetic.

A hot air balloon centered in the composition with three small clouds surrounding it. The balloon features vertical stripes in yellow-green, pale yellow, light green, and lime colors, creating a segmented appearance. Below the balloon is a small orange basket connected by thin yellow lines.

| | | |
|---|---|---|
| | | |
| | | |

A single top hat rendered in isometric perspective, featuring a tall cylindrical crown in light blue with a yellow five-pointed star on the front, a pink ribbon band, and a wide circular brim in dark gray.

###### (a) Qualitative comparison on Text-to-SVG generation.

Claude-Sonnet-4.5Gemini-3-Pro Qwen3-VL-235B-A22BInternSVG-8B CTRL-S(Ours)

GT

GPT-5.2

[Figure 2]

Three vertical binder spines stand upright on a horizontal rectangular base: the left spine is red, the center is yellow, and the right is blue and tilted diagonally.

[Figure 3]

A single arched door with a semicircular transom window at the top, featuring four rectangular brown panels arranged in two rows of two, separated by orange vertical and horizontal muntins.

| | |
|---|---|
| | |

[Figure 4]

A stylized potted plant with a pink heart-shaped bloom at the top, two symmetrical green leaves on either side, a vertical black stem, and an orange terracotta pot at the base.

###### (b) Qualitative comparison on Image-to-SVG generation.

Llama-4-MaverickQwen3-VL-235B-A22BCTRL-S(Ours)

Claude-Sonnet-4.5 Gemini-3-Pro GPT-5.2

Draft

GT

A pencil on the left and a mug on the right, both in front view. The pencil features a pink rounded eraser top, orange barrel, and black pointed tip, composed of simple rounded rectangles and a triangle.

[Figure 5]

A single rocket ship in front view, centered, with a light blue elongated teardrop body, two yellow circular windows, and two symmetrical purpleblue fins extending from the base with yellow triangular cutouts.

[Figure 6]

A flashlight in a diagonal orientation angled from lower-left to upper-right, emitting three yellow light rays. The flashlight body is gray-purple with a circular yellow button, constructed from rounded rectangles and curved paths.

[Figure 7]

(c) Qualitative comparison on SVG code refinement.

- Figure 3: Qualitative comparisons of SVG generation and code refinement between baselines and CTRL-S.

GT SFT Step-20 Step-60 Step-120 Step-200

[Figure 8]

A front-view boombox with two circular speaker grills, a rectangular cassette deck, an antenna extending diagonally upward-right with a circular tip, and a smartphone below emitting curved wireless signal lines on both sides.

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |

[Figure 9]

A two-tiered birthday cake with three lit candles on top, viewed from the front. The cake features rectangular layers in coral-red and darker red with wavy black frosting lines, cylindrical tan candles, and teardrop-shaped flames in yellow, orange, and red.

[Figure 10]

Two identical bottles with dark caps and beige bodies, each containing a pink-red liquid filling the lower twothirds, rest on a pink horizontal platform.

Figure 4: Qualitative visualization of SVG generation quality across RL training steps.

layouts, more faithful color reproduction, and richer fine-grained details, demonstrating the effectiveness of our multi-reward optimization in iteratively refining the model’s generation capability.

##### 5.4. Ablation Studies

Benefits of Chain-of-Thought Reasoning. To evaluate the impact of CoT reasoning on model performance, we construct a non-CoT dataset by directly using the original SVG codes from ColorSVG without reconstruction by Claude-Sonnet-4.5. Based on this dataset, we train baseline SFT and RL models without CoT capabilities. As shown in Table 3a, for the SFT models, introducing CoT reasoning substantially improves the task success rate. This indicates that incorporating explicit reasoning enhances generation robustness, enabling the model to produce more renderable and syntactically valid SVG programs. Moreover, for both SFT and RL settings, the CoT-enabled models consistently outperform their non-CoT counterparts across all visual quality metrics. These results suggest that CoT reasoning plays a critical role not only in improving generation stability but also in enhancing visual fidelity and semantic alignment.

Ablation on Multi-Reward Functions. To systematically investigate the specific contributions of individual reward functions, we conduct an ablation study on the reward mechanism, sequentially comparing the performance of models trained with: (1) only the Format and DINO Rewards, (2) the addition of the Image-Text Similarity Reward, and (3) the full composite reward. As demonstrated in Table 3b, for the Text-to-SVG task, incorporating the Image-Text Similarity Reward significantly enhances the model’s semantic comprehension and text-image alignment capabilities, improving the CLIP-T2I score from 24.573 to 25.444 and the CLIP-I2I score from 80.897 to 82.070. More importantly, with the further integration of the Code Efficiency Reward, the model effectively mitigates the issue of code redundancy. It not only maintains superior image generation quality but also significantly accelerates the per-sample inference speed, reducing the generated tokens from 701 to 346 and the inference time from 7.121 to 4.439 seconds per SVG. Ultimately, this achieves an optimal balance between visual fidelity and inference efficiency.

Ablation on Reward Ratio. As demonstrated in Table 3c, we empirically evaluate different weight ratios among 𝑤dino, 𝑤lclip, and 𝑤eff. Our findings indicate that the 2:1:1 configuration achieves the most effective balance among visual quality, semantic alignment, and code efficiency. Specifically, increasing 𝑤dino favors structural consistency but leads to longer outputs, while reducing it harms geometric fidelity. We therefore adopt 2:1:1 in all experiments.

Effects of Multi-task Training. We analyze the impact of joint training across Text-to-SVG, Image-to-SVG, and SVG code refinement tasks. As shown in Table 3d, compared with single-task training, multi-task learning significantly improves the overall quality of generated SVG. Jointly training Text-to-SVG and Image-to-SVG enhances cross-modal semantic alignment, leading to better consistency between textual instructions, input images, and generated SVG. Furthermore, incorporating the SVG code refinement task brings substantial gains to the Image-to-SVG setting. This suggests that learning to analyze rendered images from imperfect SVG and

Table 3: Ablation studies on CoT, reward signals, reward ratios, and multi-task training.

(a) Ablation on Chain-of-Thought (CoT).

|Model|Text-to-SVG FID ↓ CLIP-T2I ↑ CLIP-I2I ↑ SR Tokens<br><br>|Image-to-SVG DINO ↑ SSIM ↑ LPIPS ↓ SR Tokens|
|---|---|---|
|w/o CoT (SFT) w/ CoT (SFT)<br><br>|22.960 23.002 76.567 85.75% 1.0k 18.154 23.655 77.959 92.02% 1.1k<br><br>|0.887 0.663 0.204 81.96% 1.3k 0.903 0.717 0.187 90.12% 1.4k<br><br>|
|w/o CoT (SFT+RL) w/ CoT (SFT+RL)<br><br>|12.161 24.278 80.604 99.47% 274 11.584 25.944 82.291 99.85% 346<br><br>|0.961 0.813 0.125 99.53% 466 0.980 0.835 0.098 99.93% 512<br><br>|

(b) Ablation on multi-reward functions.

|Reward|FID ↓ CLIP-T2I ↑ CLIP-I2I ↑ Tokens Time (s/svg)<br><br>|
|---|---|
|𝑅format · 𝑅dino 𝑅format · (𝑅dino + 𝑅lclip) Full 𝑅total (Ours)<br><br>|12.443 24.573 80.897 606 6.151 11.831 25.444 82.070 701 7.121 11.584 25.944 82.291 346 4.439<br><br>|

(c) Ablation on reward ratio (𝑤dino:𝑤lclip:𝑤eff).

|Ratio|Text-to-SVG FID ↓ CLIP-T2I ↑ CLIP-I2I ↑ Tokens<br><br>|Image-to-SVG DINO ↑ SSIM ↑ LPIPS ↓ Tokens|
|---|---|---|
|1:1:1 5:3:1 2:1:1 (Ours)<br><br>|12.575 24.623 80.849 333 11.642 24.653 81.336 428 11.584 25.944 82.291 346<br><br>|0.962 0.818 0.120 532 0.967 0.830 0.109 649 0.980 0.835 0.098 512<br><br>|

(d) Ablation on multi-task training across Text-to-SVG, Image-to-SVG, and Refinement tasks. T2S denotes Text-to-SVG, I2S denotes Image-to-SVG, and Refine denotes SVG code refinement.

|Training<br><br>|Text-to-SVG FID ↓ CLIP-T2I ↑ CLIP-I2I ↑|Image-to-SVG DINO ↑ SSIM ↑ LPIPS ↓<br><br>|Refinement DINO ↑ SSIM ↑ LPIPS ↓<br><br>|
|---|---|---|---|
|T2S Only I2S Only Refine Only T2S + I2S Multi-Task (Ours)<br><br>|12.321 24.431 80.651<br><br>– – –<br><br>– – –<br><br><br>11.637 24.804 81.433 11.584 25.944 82.291<br><br>|– – – 0.959 0.824 0.119 – – – 0.966 0.824 0.114 0.980 0.835 0.098<br><br>|– – –<br><br>– – –<br><br><br>0.940 0.753 0.187 – – – 0.951 0.765 0.180<br><br>|

perform targeted SVG correction strengthens the model’s ability to reconstruct vector graphics from images. Overall, these results indicate that the three tasks provide complementary supervision signals, and their joint optimization leads to more robust and generalized SVG modeling.

### 6. Conclusion

In this work, we present CTRL-S, a unified framework that introduces chain-of-thought (CoT) reasoning into Text-to-SVG, Image-to-SVG, and SVG code refinement tasks. By leveraging the hierarchical grouping structure inherent in SVG, we explicitly align planning steps with code groups, thereby improving the structural consistency, readability, and editability of generated SVG. Building upon this design, we propose a multi-task, multi-reward GRPO training paradigm that jointly optimizes complementary tasks under diverse reward signals. Through cross-task collaboration and multi-perspective reward guidance, our approach significantly enhances visual fidelity, semantic alignment, and generation stability. In addition, we construct the SVG-Sophia dataset, a high-quality SVG corpus augmented with explicit CoT question–answer pairs, providing systematic training resources for structured SVG generation and code refinement. Extensive experimental results demonstrate that CTRL-S consistently achieves state-of-the-art performance across multiple tasks, validating the effectiveness of structured reasoning and multi-task, multi-reward optimization for vector graphic modeling. Overall, this work establishes a new training paradigm for reliable reasoning in SVG-LLMs and lays the foundation for future research in complex vector graphic generation and editing.

### Acknowledgements

This work was supported by the Shanghai Artificial Intelligence Laboratory and the Shanghai Committee of Science and Technology (No. 22YF1461500).

### References

- [1] Anthropic. Introducing claude sonnet 4.5. https://www.anthropic.com/news/ claude-sonnet-4-5, 2025. 3, 3.2
- [2] Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, et al. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631, 2025. 1, 3.2
- [3] Alexandre Carlier, Martin Danelljan, Alexandre Alahi, and Radu Timofte. Deepsvg: A hierarchical generative network for vector graphics animation. Advances in Neural Information Processing Systems, 33:16351–16361, 2020. 2
- [4] Hanqi Chen, Zhongyin Zhao, Ye Chen, Zhujin Liang, and Bingbing Ni. Svgthinker: Instruction-aligned and reasoning-driven text-to-svg generation. In Proceedings of the 33rd ACM International Conference on Multimedia, pages 11004–11012, 2025. 1, 2
- [5] Siqi Chen, Xinyu Dong, Haolei Xu, Xingyu Wu, Fei Tang, Hang Zhang, Yuchen Yan, Linjuan Wu, Wenqi Zhang, Guiyang Hou, et al. Svgenius: Benchmarking llms in svg understanding, editing and generation. In Proceedings of the 33rd ACM International Conference on Multimedia, pages 13289–13296, 2025.
- [6] Zehao Chen and Rong Pan. Svgbuilder: Component-based colored svg generation with text-guided autoregressive transformers. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pages 2358–2366, 2025. 2, 3, 3.2
- [7] Kevin Frans, Lisa Soros, and Olaf Witkowski. Clipdraw: Exploring text-to-drawing synthesis through language-image encoders. Advances in Neural Information Processing Systems, 35:5207–5218, 2022. 2
- [8] Google DeepMind. Gemini 3 pro model card. https://storage.googleapis.com/ deepmind-media/Model-Cards/Gemini-3-Pro-Model-Card.pdf, 2025. 3.2
- [9] David Ha and Douglas Eck. A neural representation of sketch drawings. arXiv preprint arXiv:1704.03477,

2017. 2

- [10] Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card. arXiv preprint arXiv:2410.21276,

2024. 1

- [11] Ajay Jain, Amber Xie, and Pieter Abbeel. Vectorfusion: Text-to-svg by abstracting pixel-based diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1911–1920, 2023. 2
- [12] Jinke Li, Jiarui Yu, Chenxing Wei, Hande Dong, Qiang Lin, Liangjing Yang, Zhicai Wang, and Yanbin Hao. Unisvg: A unified dataset for vector graphic understanding and generation with multimodal large language models. In Proceedings of the 33rd ACM International Conference on Multimedia, pages 13156–13163,

2025. 2

- [13] Tzu-Mao Li, Michal Lukáč, Michaël Gharbi, and Jonathan Ragan-Kelley. Differentiable vector graphics rasterization for editing and learning. ACM Transactions on Graphics (TOG), 39(6):1–15, 2020. 2
- [14] Raphael Gontijo Lopes, David Ha, Douglas Eck, and Jonathon Shlens. A learned representation for scalable vector graphics. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 7930–7939, 2019. 2
- [15] Xu Ma, Yuqian Zhou, Xingqian Xu, Bin Sun, Valerii Filev, Nikita Orlov, Yun Fu, and Humphrey Shi. Towards layer-wise image vectorization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 16314–16323, 2022. 2
- [16] Meta AI. Llama 4 maverick 17b-128e (model card). https://huggingface.co/meta-llama/ Llama-4-Maverick-17B-128E, 2025. 1

- [17] Meta AI. Llama 4 scout 17b-16e (model card). https://huggingface.co/meta-llama/ Llama-4-Scout-17B-16E, 2025. 1
- [18] OpenAI. Introducing gpt 5.2. https://openai.com/index/introducing-gpt-5-2/, 2025. 3.2
- [19] Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023. 4.3
- [20] Ben Poole, Ajay Jain, Jonathan T Barron, and Ben Mildenhall. Dreamfusion: Text-to-3d using 2d diffusion. arXiv preprint arXiv:2209.14988, 2022. 2
- [21] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PmLR,

2021. 4.3

- [22] Pradyumna Reddy, Michael Gharbi, Michal Lukac, and Niloy J Mitra. Im2vec: Synthesizing vector graphics without vector supervision. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 7342–7351, 2021. 2
- [23] Leo Sampaio Ferraz Ribeiro, Tu Bui, John Collomosse, and Moacir Ponti. Sketchformer: Transformerbased representation for sketched structure. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 14153–14162, 2020. 2
- [24] Juan A Rodriguez, Abhay Puri, Shubham Agarwal, Issam H Laradji, Pau Rodriguez, Sai Rajeswar, David Vazquez, Christopher Pal, and Marco Pedersoli. Starvector: Generating scalable vector graphics code from images and text. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 16175–16186, 2025. 1, 2
- [25] Juan A Rodriguez, Haotian Zhang, Abhay Puri, Rishav Pramanik, Aarash Feizi, Pascal Wichmann, Arnab Kumar Mondal, Mohammad Reza Samsami, Rabiul Awal, Perouz Taslakian, et al. Rendering-aware reinforcement learning for vector graphics generation. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025. 1, 2, 4.3
- [26] Peter Schaldenbrand, Zhixuan Liu, and Jean Oh. Styleclipdraw: Coupling content and style in text-todrawing translation. arXiv preprint arXiv:2202.12362, 2022. 2
- [27] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024. 1, 2, 4.1
- [28] I-Chao Shen and Bing-Yu Chen. Clipgen: A deep generative model for clipart vectorization and synthesis. IEEE Transactions on Visualization and Computer Graphics, 28(12):4211–4224, 2021. 2
- [29] Yiren Song, Xuning Shao, Kang Chen, Weidong Zhang, Zhongliang Jing, and Minzhe Li. Clipvg: Textguided image manipulation using differentiable vector graphics. In Proceedings of the AAAI conference on artificial intelligence, volume 37, pages 2312–2320, 2023. 2
- [30] Yael Vinker, Yuval Alaluf, Daniel Cohen-Or, and Ariel Shamir. Clipascene: Scene sketching with different types and levels of abstraction. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4146–4156, 2023.
- [31] Yael Vinker, Ehsan Pajouheshgar, Jessica Y Bo, Roman Christian Bachmann, Amit Haim Bermano, Daniel Cohen-Or, Amir Zamir, and Ariel Shamir. Clipasso: Semantically-aware object sketching. ACM Transactions on Graphics (TOG), 41(4):1–11, 2022. 2
- [32] Feiyu Wang, Zhiyuan Zhao, Yuandong Liu, Da Zhang, Junyu Gao, Hao Sun, and Xuelong Li. Svgen: Interpretable vector graphics generation with large language models. In Proceedings of the 33rd ACM International Conference on Multimedia, pages 9608–9617, 2025. 1, 2

- [33] Haomin Wang, Jinhui Yin, Qi Wei, Wenguang Zeng, Lixin Gu, Shenglong Ye, Zhangwei Gao, Yaohui Wang, Yanting Zhang, Yuanqi Li, Yanwen Guo, Wenhai Wang, Kai Chen, Yu Qiao, and Hongjie Zhang. Internsvg: Towards unified svg tasks with multimodal large language models. In The Fourteenth International Conference on Learning Representations, 2026. 1, 2, 4.2, 5.2
- [34] Weiyun Wang, Zhangwei Gao, Lixin Gu, Hengjun Pu, Long Cui, Xingguang Wei, Zhaoyang Liu, Linglin Jing, Shenglong Ye, Jie Shao, et al. Internvl3. 5: Advancing open-source multimodal models in versatility, reasoning, and efficiency. arXiv preprint arXiv:2508.18265, 2025. 1
- [35] Ronghuan Wu, Wanchao Su, Kede Ma, and Jing Liao. Iconshop: Text-guided vector icon synthesis with autoregressive transformers. ACM Transactions on Graphics (TOG), 42(6):1–14, 2023. 2
- [36] Ximing Xing, Yandong Guan, Jing Zhang, Dong Xu, and Qian Yu. Reason-svg: Hybrid reward rl for aha-moments in vector graphics generation. arXiv preprint arXiv:2505.24499, 2025. 1, 2
- [37] Ximing Xing, Juncheng Hu, Guotao Liang, Jing Zhang, Dong Xu, and Qian Yu. Empowering llms to understand and generate complex vector graphics. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 19487–19497, 2025. 1, 2
- [38] Ximing Xing, Chuang Wang, Haitao Zhou, Jing Zhang, Qian Yu, and Dong Xu. Diffsketcher: Text guided vector sketch synthesis through latent diffusion models. Advances in Neural Information Processing Systems, 36:15869–15889, 2023. 2
- [39] Ximing Xing, Qian Yu, Chuang Wang, Haitao Zhou, Jing Zhang, and Dong Xu. Svgdreamer++: Advancing editability and diversity in text-guided svg generation. IEEE transactions on pattern analysis and machine intelligence, 2025.
- [40] Ximing Xing, Haitao Zhou, Chuang Wang, Jing Zhang, Dong Xu, and Qian Yu. Svgdreamer: Text guided svg generation with diffusion model. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 4546–4555, 2024. 2
- [41] Yiying Yang, Wei Cheng, Sijin Chen, Xianfang Zeng, Fukun Yin, Jiaxu Zhang, Liao Wang, Gang YU, Xingjun Ma, and Yu-Gang Jiang. Omnisvg: A unified scalable vector graphics generation model. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025. 1, 2
- [42] Beichen Zhang, Pan Zhang, Xiaoyi Dong, Yuhang Zang, and Jiaqi Wang. Long-clip: Unlocking the long-text capability of clip. In European conference on computer vision, pages 310–325. Springer, 2024. 4.3
- [43] Peiying Zhang, Nanxuan Zhao, and Jing Liao. Text-to-vector generation with neural path representation. ACM Transactions on Graphics (TOG), 43(4):1–13, 2024. 2
- [44] Jinguo Zhu, Weiyun Wang, Zhe Chen, Zhaoyang Liu, Shenglong Ye, Lixin Gu, Hao Tian, Yuchen Duan, Weijie Su, Jie Shao, et al. Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models. arXiv preprint arXiv:2504.10479, 2025. 1
- [45] Bocheng Zou, Mu Cai, Jianrui Zhang, and Yong Jae Lee. Vgbench: A comprehensive benchmark of vector graphics understanding and generation for large language models. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 3647–3659, 2024. 2

### A. Appendix

##### A.1. Design of SVG-specific tokens

In this section, we summarize the design of SVG-specific tokens adopted in CTRL-S. As mentioned in the main paper, all SVGs are first mapped to a normalized 128 × 128 coordinate space. This unified parameterization reduces unnecessary variation in the absolute coordinate scale, simplifies geometric modeling, and improves consistency across different SVG-related tasks.

To better accommodate the unique elements of SVG code, we augment the original Qwen3-VL tokenizer with a dedicated set of SVG symbols. These added tokens are designed to absorb frequent multi-character patterns that would otherwise be fragmented into long subword sequences. As listed in Table 4, the vocabulary includes 49 tag-level tokens that cover common structural and graphical elements, such as <svg, <path, and <circle. We further introduce 35 attribute-level tokens for geometric fields and style declarations, including tokens like stroke=", class=", d=", and fill=".

In addition to symbolic tokens, we explicitly allocate numeric tokens for compact coordinate and parameter prediction. Specifically, the vocabulary contains 247 integer tokens ranging from −128 to 128, together with 100 two-decimal fractional tokens and 10 one-decimal fractional tokens. This numeric design allows the model to express geometric quantities more directly, while shortening output sequences and reducing decoding overhead.

The embeddings of the newly added tokens are initialized using a subword-based initialization strategy. For each added token, we first decompose it using the original tokenizer and then set its initial embedding as the average of the corresponding subword embeddings. This strategy provides a smooth initialization for all SVG-specific tokens, preserving compatibility with the pretrained embedding space. After expansion, all parameters are optimized jointly in an end-to-end fashion. In practice, this initialization scheme improves training stability and helps the model more reliably generate structurally valid SVG code and numerically accurate parameters.

Table 4: SVG-specific tokens used in CTRL-S.

(a) Tag tokens

|Category|Tokens<br><br>|
|---|---|
|Root|<svg, </svg>, <defs, </defs>, <use, </use> ,/>|
|Grouping<br><br>|<g, </g>|
|Shapes<br><br>|<path, </path>, <rect, </rect>, <circle, </circle>, <ellipse, </ellipse>, <line, </line>, <polyline, </polyline>, <polygon, </polygon>|
|Text<br><br>|<text, </text>, <tspan, </tspan>, <textPath, </textPath>|
|Gradients<br><br>|<linearGradient, </linearGradient>, <radialGradient, </radialGradient>, <stop, </stop>|
|Clipping<br><br>|<clipPath, </clipPath>, <mask, </mask>|
|Filters<br><br>|<filter, </filter>, <feGaussianBlur, </feGaussianBlur>, <feColorMatrix, </feColorMatrix>, <feComposite, </feComposite>, <feBlend, </feBlend>|

(b) Attribute tokens

|Category|Tokens<br><br>|
|---|---|
|Geometry<br><br>|width=", height=", viewBox=", x=", y=", x1=", y1=", x2=", y2=", cx=", cy=", r=", rx=", ry=", d=", points="|
|Styling|fill=", stroke=", stroke-width=", stroke-linecap=", stroke-linejoin=", stroke-miterlimit=", fill-rule=", opacity="<br><br>|
|Transform|transform="<br><br>|
|Text<br><br>|font-size=", font-family=", text-anchor="|
|Gradients|gradientUnits=", gradientTransform=", offset=", stop-color="|
|Identifiers|id=", class=", clip-path="|

##### A.2. Examples of SVG-Sophia

To provide a more intuitive understanding of SVG-Sophia, this section presents representative examples of Text-to-SVG, Image-to-SVG, and SVG code refinement tasks. Specifically, for the Text-to-SVG task, we provide detailed instructions to guide the model in generating high-quality SVGs. For the Image-to-SVG task, we supplement these instructions with raster images to facilitate accurate vectorization. For the SVG code refinement task, we supply the rasterized target image, instructions, and the currently defective SVG code to guide the model in code repair. As illustrated in Figures 5, 6, and 7, each sample features structured reasoning alongside its corresponding SVG code, demonstrating the diverse visual concepts and code patterns covered by SVG-Sophia. These examples highlight the high-quality annotations within SVG-Sophia and showcase how the dataset facilitates both structured reasoning and fine-grained SVG generation and code refinement.

<think> Canvas: 128 x 128

You are a master of SVG development. Analyze the image and generate an SVG with the required format: <think>

Image Description: A bowl of food with two chopsticks resting on top, viewed from the front. The bowl is depicted as a yellow semicircle with a scalloped cream-colored layer representing rice or foam on top. Two coral-red elongated rounded rectangles represent chopsticks angled upward to the right. The design uses smooth curves, rounded edges, flat fills in yellow, cream, and coral-red colors, with thick solid black strokes and round caps and joins. The composition is centered and symmetrical, with the chopsticks layered above the bowl and rice.

- - Canvas dimensions
- - Image description
- - Step-by-step module breakdown

</think> ```svg [Modular SVG code with descriptive comments] ``` Let's think step by step and generate the SVG based on the instruction below. Instruction: A bowl of food with two chopsticks resting on top, viewed from the front. The bowl is a yellow semicircle with a scalloped cream-colored layer representing rice or foam, and two coral-red elongated rounded rectangles for chopsticks angled upward to the right. The vector uses smooth curves, rounded edges, flat fills in yellow, cream, and coral-red, with thick solid black strokes and round caps and joins. The composition is centered and symmetrical, with the chopsticks layered above the bowl and rice.

Drawing Steps:

- 1. Black outline/shadow layer - Creates the dark background outline and shadow effect for the entire composition
- 2. First chopstick (lower) - Coral-red chopstick positioned at a lower angle
- 3. Yellow bowl - The main bowl body rendered as a semicircle
- 4. Cream rice/foam layer - Scalloped decorative layer on top of the bowl representing rice or foam
- 5. Second chopstick (upper) - Coral-red chopstick positioned at a higher angle, overlapping the bowl </think>

SVG

<think> Canvas: 128 x 128

You are a professional SVG developer. Please analyze the image and create an SVG following this structure: <think>

Image Description: Four square icons arranged in a 2×2 grid layout. Top-left icon shows a

- - Canvas size
- - Overall image description
- - Numbered drawing steps

landscape scene with a yellow sun (circle) above an orange/brown mountain (triangle) on a light blue sky background. Top-right and bottom-left icons display horizontal black lines representing text on

</think> ```svg [Well-organized modular SVG code] ``` Let's think step by step and generate the SVG based on the instruction below. Instruction: Four square icons arranged in a 2×2 grid, each with thick black rounded-rectangle borders and distinct content: top-left shows a yellow circle (sun) above an orange triangle (mountain) on light blue; topright and bottom-left display horizontal black lines (text) on white; bottom-right features a large purple capital "T" on white. The vector uses simple geometric primitives—circles, triangles, rectangles, and rounded paths—with thick solid black strokes, round caps and joins, and flat fills in yellow, orange, light blue, purple, white, and black. All four icons are evenly spaced and aligned in a symmetrical grid layout.

white backgrounds. Bottom-right icon features a large purple capital letter "T" on a white background. All icons have thick black roundedrectangle borders with consistent styling.

Drawing Steps:

- 1. Top-left icon background (light blue sky with darker blue accent)
- 2. Top-left icon sun (yellow circle with orange shading)
- 3. Top-left icon mountain (orange/brown triangle with darker shading)
- 4. Bottom-left icon background (white with gray accent)
- 5. Top-right icon background (white with gray accent)
- 6. Bottom-right icon background (white with gray accent)
- 7. Bottom-right icon letter T (purple with darker purple shading)
- 8. All borders and outlines (black strokes for all four icons)
- 9. Bottom-left icon text lines (horizontal black lines)
- 10. Top-right icon text lines (horizontal black lines)
- 11. Bottom-right icon letter T outline (black border) </think>

SVG

Figure 5: Examples of Text-to-SVG in SVG-Sophia.

<think> Canvas: 128 x 128

You are a skilled SVG designer. Your task is to convert the image into an SVG with the following format: <think>

Image Description: A three-tiered decorative fountain icon displayed in a symmetrical front view. The fountain consists of three stacked levels, each with a wide shallow basin and supporting pedestal, decreasing in size from bottom to top. The design uses simple geometric shapes with rounded corners, rendered in a flat style with light gray (#e4e8eb), medium gray (#cad0d7), and blue-gray (#a9b0bb) colors to create subtle depth through layered shapes.

- - Canvas dimensions (from viewBox)
- - Description of what the image depicts
- - Sequential drawing steps

</think> ```svg [Modular SVG with module comments] ``` Let's think step by step and generate the SVG based on the provided image and instruction. Image: <image> Instruction: A three-tiered fountain icon composed of stacked rounded bowls and pedestals, viewed from the front in symmetrical arrangement. Each tier features a wide, shallow basin with rounded edges atop a trapezoidal pedestal, decreasing in size from bottom to top. The vector uses simple geometric primitivesrounded rectangles and curved paths—with smooth, rounded corners and bilateral symmetry. Flat fills in light gray and blue-gray tones create subtle depth through layered shapes, with no visible strokes. The composition is centered and vertically aligned, with each tier directly above the next.

Drawing Steps:

- 1. Top tier water spout - Small vertical element at the very top where water would emerge
- 2. Top tier basin - Smallest basin (third level) with its pedestal support
- 3. Top tier rim - Decorative edge/lip of the top basin
- 4. Middle tier water spout - Vertical element connecting top to middle tier
- 5. Middle tier basin - Medium-sized basin (second level) with its pedestal
- 6. Middle tier rim - Decorative edge/lip of the middle basin
- 7. Bottom tier water spout - Vertical element connecting middle to bottom tier
- 8. Bottom tier basin - Largest basin (bottom level) with its pedestal
- 9. Bottom tier rim - Decorative edge/lip of the bottom basin
- 10. Base platform - Foundation platform supporting the entire fountain
- 11. Base rim - Decorative edge/lip of the base platform </think>

SVG

You are a professional SVG developer. Please analyze the image and create an SVG following this structure: <think>

<think> Canvas: 128 x 128

- - Canvas size
- - Overall image description
- - Numbered drawing steps

Image Description: A steaming cup of coffee or tea viewed from the front. The cup features a light blue body with a darker blue outline and handle on the right side. Three wavy steam lines rise vertically above the cup. The design uses a two-tone blue color palette with flat fills and rounded stroke styles. Decorative dots and horizontal lines accent the cup's rim and body, suggesting liquid level. The composition is centered with symmetrical geometry.

</think> ```svg [Well-organized modular SVG code] ``` Let's think step by step and generate the SVG based on the provided image and instruction. Image: <image> Instruction: A steaming cup viewed from the front, featuring three wavy steam lines rising above a rounded mug with a handle on the right side. The cup body is filled with flat light blue, outlined by thick rounded blue strokes; small circular dots accent the rim and base, and horizontal bars suggest liquid level. The vector uses smooth curves, rounded caps and joins, symmetrical geometry for the cup body, and a two-tone blue palette with flat fills and no gradients. The composition centers the cup with steam aligned vertically above and the handle extending rightward.

Drawing Steps:

- 1. Cup body fill - Light blue rounded shape forming the main body of the cup
- 2. Cup rim decorative elements - Horizontal lines and dots at the top edge of the cup
- 3. Cup body left side details - Dots and decorative elements on the left side
- 4. Cup body right side details - Dots and decorative elements on the right side
- 5. Cup liquid level indicators - Horizontal lines suggesting the liquid inside
- 6. Cup body outline and structure - Main dark blue outline defining the cup shape and base
- 7. Cup handle - Dark blue handle extending to the right side
- 8. Steam lines - Three wavy vertical lines rising above the cup (left, middle, right) </think>

SVG

Figure 6: Examples of Image-to-SVG in SVG-Sophia.

As a professional vector graphics editor, your task is to refine the draft SVG. Analyze the Reference Image and the Draft Code, then generate the correct SVG. Response Format: <think>

<think> Canvas: 128 x 128

Discrepancy Analysis:

- - Point 1: The door in the Draft is positioned on the left side and colored bright orange (#FF6B35), whereas in the Reference it is centered and colored in a lighter peachy-orange tone (#F4B183).
- - Point 2: The Draft has only two windows (one large rectangular window on the upper right and one small window on the right side), while the Reference has three light blue windows - two on the left side of the main body and one on the door itself.
- - Point 3: The coral-red taillight accents in the Draft appear as single rectangular elements on both sides, but in the Reference they are positioned differently with a two-tone effect (lighter coral above darker coral-red) on the left rear.

- - Canvas Analysis
- - Discrepancy Analysis (Draft vs Reference)
- - Refinement Steps </think> ```svg [Final SVG Code] ```

Image

Let's think step by step. Image: <image> Instruction: A yellow recreational vehicle in side view with two large black wheels featuring light gray hubs, an orange door with a light blue window, and additional light blue windows on the body. The vector uses simple rounded rectangles and circles with flat fills in yellow, orange, light blue, black, gray, and coral-red accents for taillights, with no visible strokes. The composition is symmetrical along the horizontal axis, with the door centered and wheels evenly spaced, creating a clean, minimalist flatdesign aesthetic. Draft Code:

Refinement Steps:

- 1. Reposition the orange door element from the left side to the center of the RV body and adjust its color to a lighter peachy-orange (#F4B183 or similar).
- 2. Add a third light blue window on the door element itself, and reposition the existing windows to match the Reference layout (two windows on the left portion of the main body).
- 3. Adjust the taillight elements to create the two-tone coral effect on the rear left side, matching the Reference's color scheme and positioning.
- 4. Verify the door window is vertically centered within the door element and properly sized according to the Reference proportions. </think>

SVG

Draft SVG

<think> Canvas: 128 x 128

As a professional vector graphics editor, your task is to refine the draft SVG. Analyze the Reference Image and the Draft Code, then generate the correct SVG.

Discrepancy Analysis:

- - Point 1: The pink body shape differs significantly - the Reference shows a wide, flattened rounded oval (wider than tall), while the Draft shows a more circular/spherical shape that is taller and narrower.
- - Point 2: The green calyx/stem positioning and size are incorrect - in the Reference it's a large three-lobed element sitting on top and overlapping the pink body, while in the Draft it appears smaller, more angular/geometric, and positioned differently with less organic curves.
- - Point 3: The Draft contains an extra lighter pink oval element in the center that is not present in the Reference design, violating the simple two-layer composition specified in the instruction.

Response Format: <think>

- - Canvas Analysis
- - Discrepancy Analysis (Draft vs Reference)
- - Refinement Steps </think> ```svg [Final SVG Code] ```

Let's think step by step. Image: <image> Instruction: A stylized tomato composed of two overlapping shapes: a large pink rounded oval body and a smaller light green three-lobed stem or calyx on top. The vector uses smooth, organic curves with no visible strokes, filled with flat solid colors in bright pink and mint green. The composition is centered and symmetrical, with the green element layered above and slightly overlapping the pink base, creating a simple two-layer depth. Draft Code:

Image

Refinement Steps:

- 1. Modify the pink body shape to be a wider, more horizontally-oriented rounded oval (ellipse-like) rather than circular, matching the flattened proportions of the Reference.
- 2. Redesign the green calyx element to be larger with three smooth, organic lobes that properly overlap the top of the pink body, using curved paths instead of angular shapes.
- 3. Remove the extra lighter pink oval element from the center to maintain the simple two-layer composition as specified.
- 4. Ensure both elements are properly centered and the green element is layered above the pink base with appropriate overlap. </think>

SVG

Draft SVG

Figure 7: Examples of SVG code refinement in SVG-Sophia.

